# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['desktop\\client.py'],
    pathex=[],
    binaries=[],
    datas=[('D:\\DDQ\\RAGmaterial', 'RAGmaterial'), ('D:\\DDQ\\data', 'data')],
    hiddenimports=['docx', 'PyPDF2', 'openpyxl', 'pydantic', 'fastapi', 'uvicorn', 'sentence_transformers', 'transformers', 'torch', 'sklearn', 'numpy', 'scipy', 'PIL', 'tokenizers', 'huggingface_hub', 'pyarrow', 'app.config', 'app.embeddings', 'app.ingest', 'app.retrieve', 'app.generate', 'app.storage', 'app.confidence', 'app.library', 'app.excel_processor', 'app.prompts', 'app.schemas', 'app.text_cleaner', 'app.metadata_filters', 'app.ranking', 'app.reranker', 'app.evaluation', 'pandas'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['jupyter', 'notebook', 'IPython', 'matplotlib'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DDQ-RAG-Workbench',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='NONE',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DDQ-RAG-Workbench',
)
