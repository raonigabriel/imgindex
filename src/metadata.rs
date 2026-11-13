use std::io::{BufReader, Read};
use std::path::Path;

use serde_json::{Map, Value};

/// Maximum characters kept per metadata value; some tags (thumbnail
/// data, long comments) are huge blobs we don't want to store whole.
const MAX_VALUE_LEN: usize = 1024;
/// Safety caps for PNG text chunks: a file may declare arbitrarily many
/// or huge chunks, so we keep a bounded number of entries and only walk
/// a bounded prefix of the file (text chunks always precede the image
/// data in practice).
const MAX_TEXT_ENTRIES: usize = 64;
const MAX_PNG_PREFIX_BYTES: u64 = 2 * 1024 * 1024;

/// Extract image metadata and serialize it as a JSON object namespaced by
/// source. NOT assumed to be EXIF only:
///
///   {"exif":     {"Tag": "display value", ...},   // JPEG/TIFF containers and the Exif/EXIF chunks of PNG/WebP
///    "png_text": {"Keyword": "text", ...}}       // PNG tEXt / zTXt / iTXt / tIME chunks
///
/// Returns `None` when the file carries neither kind; unsupported or
/// unreadable content simply yields no metadata.
pub fn extract(path: &Path, mime_type: &str) -> Option<String> {
    let mut root = Map::new();

    let exif = exif_map(path);
    if !exif.is_empty() {
        root.insert("exif".into(), Value::Object(exif));
    }

    if mime_type == "image/png" {
        let png_text = png_text_map(path);
        if !png_text.is_empty() {
            root.insert("png_text".into(), Value::Object(png_text));
        }
    }

    if root.is_empty() {
        None
    } else {
        Some(Value::Object(root).to_string())
    }
}

/// EXIF via a kamadak-exif container read: JPEG/TIFF directly, the `eXIf`
/// chunk of PNGs, the `EXIF` chunk of WebPs. Absent/unsupported -> empty.
fn exif_map(path: &Path) -> Map<String, Value> {
    let mut map = Map::new();
    let Ok(file) = std::fs::File::open(path) else {
        return map;
    };
    let reader = exif::Reader::new();
    let Ok(container) = reader.read_from_container(&mut BufReader::new(&file)) else {
        return map;
    };
    for field in container.fields() {
        let key = field.tag.to_string();
        let mut value = field.display_value().to_string();
        if value.chars().count() > MAX_VALUE_LEN {
            value = value.chars().take(MAX_VALUE_LEN).collect();
            value.push('…');
        }
        map.insert(key, Value::String(value));
    }
    map
}

/// PNG-native text metadata: read a bounded prefix of the file and decode
/// the text/timing chunks it contains.
fn png_text_map(path: &Path) -> Map<String, Value> {
    let mut map = Map::new();
    let Ok(file) = std::fs::File::open(path) else {
        return map;
    };
    let mut data = Vec::new();
    if BufReader::new(file)
        .take(MAX_PNG_PREFIX_BYTES)
        .read_to_end(&mut data)
        .is_err()
    {
        return map;
    }
    for (key, value) in png_text_chunks(&data) {
        map.insert(key, Value::String(value));
    }
    map
}

/// Walk PNG chunks (after the 8-byte signature) and collect their
/// key/value pairs. Length fields are untrusted: every bound is checked,
/// a truncated or malformed chunk just ends the walk.
fn png_text_chunks(data: &[u8]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut pos = 8usize;
    while pos + 8 <= data.len() && out.len() < MAX_TEXT_ENTRIES {
        let len = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        let ctype = [data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]];
        let Some(body_start) = pos.checked_add(8) else {
            break;
        };
        let Some(body_end) = body_start.checked_add(len) else {
            break;
        };
        if body_end.saturating_add(4) > data.len() {
            break; // trailing CRC would overrun: treat as truncated file
        }
        let body = &data[body_start..body_end];

        match &ctype {
            // keyword\0text (Latin-1)
            b"tEXt" => {
                if let Some((keyword, rest)) = split_keyword(body) {
                    out.push((latin1(keyword), clamp(latin1(rest))));
                }
            }
            // keyword\0compression_method(0=zlib)zlib_stream (Latin-1)
            b"zTXt" => {
                if let Some((keyword, rest)) = split_keyword(body)
                    && rest.first() == Some(&0)
                    && let Some(decoded) = inflate(&rest[1..])
                {
                    out.push((latin1(keyword), clamp(latin1(&decoded))));
                }
            }
            // keyword\0compression_flag compression_method language\0translated\0text (UTF-8)
            b"iTXt" => {
                if let Some((keyword, rest)) = split_keyword(body)
                    && rest.len() >= 2
                    && let Some(text_start) = skip_nul_terminated(&rest[2..], 2)
                {
                    let compressed = rest[0] == 1;
                    let text = &rest[2 + text_start..];
                    let value = if compressed {
                        inflate(text).map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                    } else {
                        Some(String::from_utf8_lossy(text).into_owned())
                    };
                    if let Some(v) = value {
                        out.push((latin1(keyword), clamp(v)));
                    }
                }
            }
            // year(2 BE) month day hour minute second
            b"tIME" => {
                if body.len() == 7 {
                    let year = u16::from_be_bytes([body[0], body[1]]);
                    let stamp = format!(
                        "{}-{:02}-{:02} {:02}:{:02}:{:02}",
                        year, body[2], body[3], body[4], body[5], body[6]
                    );
                    out.push(("tIME".into(), stamp));
                }
            }
            b"IEND" => break,
            _ => {}
        }
        pos = body_end + 4; // skip the trailing CRC
    }
    out
}

/// Split `keyword\0rest` at the first NUL byte.
fn split_keyword(body: &[u8]) -> Option<(&[u8], &[u8])> {
    let nul = body.iter().position(|&b| b == 0)?;
    Some((&body[..nul], &body[nul + 1..]))
}

/// Advance past `count` NUL-terminated fields, returning the offset just
/// after the last NUL.
fn skip_nul_terminated(data: &[u8], count: usize) -> Option<usize> {
    let mut pos = 0;
    for _ in 0..count {
        let nul = data[pos..].iter().position(|&b| b == 0)? + pos;
        pos = nul + 1;
    }
    Some(pos)
}

/// Latin-1 is a byte-per-character encoding by construction.
fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| b as char).collect()
}

fn inflate(data: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    flate2::read::ZlibDecoder::new(data).read_to_end(&mut out).ok()?;
    Some(out)
}

fn clamp(mut value: String) -> String {
    if value.chars().count() > MAX_VALUE_LEN {
        value = value.chars().take(MAX_VALUE_LEN).collect();
        value.push('…');
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;

    fn chunk(ctype: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut v = (body.len() as u32).to_be_bytes().to_vec();
        v.extend_from_slice(ctype);
        v.extend_from_slice(body);
        // The chunk walker skips 4 bytes of CRC after every body; real
        // CRCs aren't validated, but the bytes must be present.
        v.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
        v
    }

    #[test]
    fn png_text_chunks_parses_all_chunk_types() {
        let mut zlib = ZlibEncoder::new(Vec::new(), Compression::default());
        zlib.write_all(b"compressed comment").unwrap();
        let zdata = zlib.finish().unwrap();

        let mut data = b"\x89PNG\r\n\x1a\n".to_vec();
        data.extend(chunk(b"tEXt", b"Title\x00my title"));
        data.extend(chunk(b"iTXt", b"Description\x00\x00\x00\x00\x00utf8 \xE2\x9C\x93"));
        let mut zbody = b"Comment\x00\x00".to_vec();
        zbody.extend(zdata);
        data.extend(chunk(b"zTXt", &zbody));
        data.extend(chunk(b"tIME", &[0x07, 0xEA, 11, 12, 15, 30, 0]));
        data.extend(chunk(b"IEND", b""));

        let got = png_text_chunks(&data);
        assert!(got.contains(&("Title".to_string(), "my title".to_string())));
        assert!(got.contains(&("Description".to_string(), "utf8 ✓".to_string())));
        assert!(got.contains(&("Comment".to_string(), "compressed comment".to_string())));
        assert!(got.contains(&("tIME".to_string(), "2026-11-12 15:30:00".to_string())));
    }

    #[test]
    fn png_text_chunks_survives_truncated_files() {
        // length field claims 4 GB; walk must stop cleanly
        let mut data = b"\x89PNG\r\n\x1a\n".to_vec();
        data.extend(0xFFFF_FFFF_u32.to_be_bytes());
        data.extend(b"tEXt");
        assert!(png_text_chunks(&data).is_empty());
    }
}
