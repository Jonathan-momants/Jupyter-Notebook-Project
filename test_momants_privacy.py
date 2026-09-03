"""Unit tests for privacy masking without real personal data."""

from __future__ import annotations

import unittest

from momants_privacy import mask_pii


class MaskPiiTests(unittest.TestCase):
    def test_masks_supported_phone_formats(self) -> None:
        examples = [
            "Bel 06-12345678",
            "Bel 0612345678",
            "Bel 06 12 34 56 78",
            "Bel +31 6 12345678",
            "Bel 0031612345678",
            "Bel +31(0)612345678",
            "Bel 010-1234567",
            "Bel 020 1234567",
            "Call +49 30 12345678",
            "Bel +32 470 12 34 56",
        ]
        for example in examples:
            with self.subTest(example=example):
                masked, count = mask_pii(example)
                self.assertEqual(count, 1)
                self.assertIn("[PHONE]", masked)

    def test_masks_email_addresses(self) -> None:
        masked, count = mask_pii("Mail test.person+festival@example.nl")
        self.assertEqual(masked, "Mail [EMAIL]")
        self.assertEqual(count, 1)

    def test_masks_multiple_values_and_counts_each_one(self) -> None:
        masked, count = mask_pii(
            "Mail bezoeker@example.nl of bel +31 6 12345678."
        )
        self.assertEqual(masked, "Mail [EMAIL] of bel [PHONE].")
        self.assertEqual(count, 2)

    def test_does_not_mask_non_pii_examples(self) -> None:
        examples = [
            "de rij was 10-12 personen lang",
            "we zijn open van 10-18 uur",
            "vak 4",
            "podium 2",
            "ik wacht al 40 minuten",
            "1 juli 2026",
            "24,50 euro",
            "0越6 is geen nummer",
        ]
        for example in examples:
            with self.subTest(example=example):
                self.assertEqual(mask_pii(example), (example, 0))

    def test_does_not_modify_bare_urls(self) -> None:
        for url in ["https://decibel.nl/programma", "www.decibel.nl/faq"]:
            with self.subTest(url=url):
                self.assertEqual(mask_pii(url), (url, 0))


if __name__ == "__main__":
    unittest.main()