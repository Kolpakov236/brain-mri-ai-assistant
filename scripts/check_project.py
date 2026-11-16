"""Check that a fresh checkout contains the runnable prototype and submission files."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    required = (
        'assets/workspace.css', 'models/best.pt', 'docs/STAGE2.md', 'docs/DESIGN.md',
        'docs/guide_for_friend.md', 'docs/guide_for_friend.pdf',
        'docs/NeuroAssistant_stage2.pptx', 'docs/NeuroAssistant_stage2.pdf',
        'docs/example_report.pdf', 'reports/stage2/comparison.json',
    )
    for name in required:
        path = ROOT / name
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f'Missing project material: {name}')
    for name in ('docs/guide_for_friend.pdf', 'docs/NeuroAssistant_stage2.pdf', 'docs/example_report.pdf'):
        with (ROOT / name).open('rb') as stream:
            if not stream.read(5).startswith(b'%PDF-'):
                raise RuntimeError(f'Invalid PDF: {name}')
    with zipfile.ZipFile(ROOT / 'docs/NeuroAssistant_stage2.pptx') as z:
        if z.testzip() is not None:
            raise RuntimeError('Presentation archive is corrupted')
        slides = [name for name in z.namelist() if name.startswith('ppt/slides/slide') and name.endswith('.xml')]
        if len(slides) != 21:
            raise RuntimeError(f'Expected the complete 21-slide presentation, found {len(slides)}')
        for name in slides:
            ElementTree.fromstring(z.read(name))
    print('Submission files: OK')

    from app import analyze, build_ui
    sample = ROOT / 'examples/example_glioma.jpg'
    gallery, cards, explanation = analyze([str(sample)])
    if len(gallery) != 2 or sample.name not in cards or not explanation.strip():
        raise RuntimeError('The saved model did not produce a complete result')
    print('Saved checkpoint and CPU inference: OK')
    ui = build_ui()
    if not ui.blocks:
        raise RuntimeError('Gradio workspace did not build')
    print('Gradio workspace: OK')


if __name__ == '__main__':
    main()
