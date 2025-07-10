from __future__ import annotations

import re
import mwparserfromhell as mwp


# Optional dependencies ------------------------------------------------------
try:
    from pint import UnitRegistry  # unit conversion

    _ureg = UnitRegistry()
except ImportError:
    _ureg = None


def _strip_templates(code: mwp.wikicode.Wikicode) -> None:
    for tpl in list(code.filter_templates(recursive=False)):
        code.remove(tpl)


def _strip_external_brackets(text: str) -> str:
    return re.sub(r"\[citation needed\]", "", text, flags=re.I)


def _convert_units_si(text: str) -> str:
    if _ureg is None:
        return text

    def _repl(match: re.Match) -> str:
        num, unit = match.groups()
        try:
            qty = _ureg.Quantity(float(num.replace(",", "")), unit)
            qty_si = qty.to_base_units()
            return f"{qty_si.magnitude:g} {qty_si.units}"
        except Exception:
            return match.group(0)

    pattern = r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(km|m|kg|g|km/s|m/s)"
    return re.sub(pattern, _repl, text)


def parse_wikitext(raw_text: str) -> str:
    code = mwp.parse(raw_text)
    _strip_templates(code)
    plain = code.strip_code()
    plain = _strip_external_brackets(plain)
    plain = _convert_units_si(plain)
    plain = re.sub(r"\n{3,}", "\n\n", plain)
    return plain.strip()
