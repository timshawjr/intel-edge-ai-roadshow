import unittest
from tools.verify_sources import looks_like_xml, looks_like_mp4, looks_like_binary, Source, Result

class ContentVerification(unittest.TestCase):
    def test_xml_must_declare_itself(self):
        self.assertTrue(looks_like_xml(b'<?xml version="1.0"?><net/>'))
        self.assertFalse(looks_like_xml(b'<net/>'))
        self.assertFalse(looks_like_xml(b'{"error": "not found"}'))

    def test_mp4_must_carry_an_ftyp_box(self):
        self.assertTrue(looks_like_mp4(b'\x00\x00\x00\x18ftypmp42' + b'\x00' * 8))
        self.assertFalse(looks_like_mp4(b'<html>404</html>'))

    def test_binary_payload_must_not_be_html_or_an_lfs_pointer(self):
        self.assertTrue(looks_like_binary(bytes(range(256)) * 4))
        self.assertFalse(looks_like_binary(b'<!DOCTYPE html><html>nope</html>'))
        self.assertFalse(looks_like_binary(b'version https://git-lfs.github.com/spec/v1\n'))
