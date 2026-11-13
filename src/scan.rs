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
