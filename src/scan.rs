use std::time::SystemTime;

use sha2::{Digest, Sha256};

/// Hex-encoded SHA-256 of a byte slice.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Convert a filesystem timestamp to unix seconds since epoch.
/// `None` when the platform doesn't provide it (e.g. birth time on old
/// Linux kernels) or it predates the epoch.
pub fn unix_ts(t: Option<SystemTime>) -> Option<i64> {
    t.and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
}

/// The machine's host name, stored with every observation row: several
/// machines may append to one shared database, and `hostname` +
/// `scanned_at` together identify which run on which box produced rows.
pub fn hostname() -> String {
    #[cfg(unix)]
    {
        // gethostname(2): writes into a fixed buffer, NUL-terminated
        // when the name fits (255 bytes is the POSIX NAME_MAX).
        let mut buf = [0u8; 256];
        let ok = unsafe {
            libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len())
        } == 0;
        if ok {
            let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            return String::from_utf8_lossy(&buf[..end]).into_owned();
        }
    }
    #[cfg(not(unix))]
    {
        if let Ok(name) = std::env::var("COMPUTERNAME") {
            return name;
        }
    }
    "unknown".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vector() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn unix_ts_handles_missing_and_pre_epoch() {
        assert_eq!(unix_ts(None), None);
        assert_eq!(
            unix_ts(Some(SystemTime::UNIX_EPOCH - std::time::Duration::from_secs(1))),
            None
        );
        assert_eq!(
            unix_ts(Some(SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(42))),
            Some(42)
        );
    }
}
