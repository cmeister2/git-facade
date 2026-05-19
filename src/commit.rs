//! Git commit object parsing.

/// Newline byte.
const NEWLINE_BYTE: u8 = 0x0a;

/// A single header line from a git commit object.
#[derive(Debug, Clone)]
pub struct Header {
    /// The full header line value (e.g. "tree abc123" or "author ...").
    pub value: String,
}

/// A parsed git commit object.
#[derive(Debug, Clone)]
pub struct Object {
    /// The object type (always "commit").
    pub object_type: String,
    /// The commit message bytes.
    pub message: Vec<u8>,
    /// The parsed headers.
    pub headers: Vec<Header>,
}

/// Parses a git commit object payload (without the `commit <len>\0` prefix).
///
/// # Errors
///
/// Returns an error if headers cannot be parsed or the commit message is empty.
pub fn parse_git_commit_object(object_payload: &[u8]) -> Result<Object, String> {
    let mut pos = 0;
    let headers = parse_headers(object_payload, &mut pos)?;
    let message = parse_commit_message(&object_payload[pos..])?;

    Ok(Object {
        object_type: "commit".to_string(),
        message,
        headers,
    })
}

/// Parses all headers from the data starting at `pos`.
fn parse_headers(data: &[u8], pos: &mut usize) -> Result<Vec<Header>, String> {
    let mut headers = Vec::new();
    loop {
        let header = parse_next_header(data, pos)?;
        match header {
            Some(h) => headers.push(h),
            None => break,
        }
    }
    Ok(headers)
}

/// Parses the next header line, returning `None` on the empty separator line.
fn parse_next_header(data: &[u8], pos: &mut usize) -> Result<Option<Header>, String> {
    let start = *pos;
    let newline_pos = data[start..]
        .iter()
        .position(|&b| b == NEWLINE_BYTE)
        .ok_or_else(|| "cannot parse commit header: no newline found".to_string())?;

    let header_bytes = &data[start..start + newline_pos];
    *pos = start + newline_pos + 1;

    if header_bytes.is_empty() {
        return Ok(None);
    }

    let value =
        String::from_utf8(header_bytes.to_vec()).map_err(|e| format!("invalid header: {}", e))?;

    Ok(Some(Header { value }))
}

/// Extracts the commit message from the remaining data after headers.
fn parse_commit_message(data: &[u8]) -> Result<Vec<u8>, String> {
    if data.is_empty() {
        return Err("empty commit message".to_string());
    }
    Ok(data.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAW_HEADER_AND_BODY_OBJECT: &str = "tree e57181f20b062532907436169bb5823b6af2f099\n\
        author Thomas Richner <thomas.richner@oviva.com> 1653693519 +0200\n\
        committer Thomas Richner <thomas.richner@oviva.com> 1653693519 +0200\n\
        \n\
        Initial commit\n\
        36abde0100000000";

    #[test]
    fn test_parse_headers() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();

        assert_eq!(obj.headers.len(), 3);
        assert!(obj.headers[0].value.starts_with("tree "));
        assert!(obj.headers[1].value.starts_with("author "));
        assert!(obj.headers[2].value.starts_with("committer "));
    }

    #[test]
    fn test_parse_message() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let msg = String::from_utf8(obj.message.clone()).unwrap();
        assert!(msg.starts_with("Initial commit"));
        assert!(msg.contains("36abde0100000000"));
    }

    #[test]
    fn test_parse_object_type() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        assert_eq!(obj.object_type, "commit");
    }

    #[test]
    fn test_parse_empty_message_fails() {
        let data = b"tree abc123\n\n";
        let result = parse_git_commit_object(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_header_values() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        assert_eq!(
            obj.headers[0].value,
            "tree e57181f20b062532907436169bb5823b6af2f099"
        );
        assert_eq!(
            obj.headers[1].value,
            "author Thomas Richner <thomas.richner@oviva.com> 1653693519 +0200"
        );
        assert_eq!(
            obj.headers[2].value,
            "committer Thomas Richner <thomas.richner@oviva.com> 1653693519 +0200"
        );
    }

    #[test]
    fn test_parse_no_newline_at_end_of_headers() {
        let data = b"tree abc\nauthor foo";
        let result = parse_git_commit_object(data);
        // "author foo" has no trailing newline, so parsing should fail
        assert!(result.is_err());
    }
}
