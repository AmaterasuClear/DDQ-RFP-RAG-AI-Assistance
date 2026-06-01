from __future__ import annotations

import faulthandler
import sys
import os

# Enable segfault tracing — writes Python stack trace to stderr on crash.
# In --windowed PyInstaller builds, stderr is None; silently skip in that case.
try:
    faulthandler.enable()
except (RuntimeError, ValueError, AttributeError):
    pass

# Enforce offline mode before any HF-adjacent imports
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Prevent torch thread-pool conflicts with Qt event loop (prevents segfaults)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Pre-load pyarrow before any other C extension to avoid PyInstaller DLL conflicts.
# sentence_transformers → sklearn → pandas → pyarrow; if pyarrow loads after
# torch/PySide6 DLLs are initialized, it causes an access violation.
try:
    import pyarrow as _  # noqa: F401
except ImportError:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.config import RAG_MATERIAL_DIR, ensure_runtime_dirs
from app.excel_processor import process_questionnaire
from app.library import delete_library_document, index_document, index_folder, stale_documents
from app.main import run_question
from app.schemas import AnswerResponse, PipelineResult
from app.storage import clear_all_reviews, list_chunks, list_documents, list_review_items, update_review_status

# ── Codex-Inspired Professional Theme ──────────────────────────────────────────
# Minimal, content-focused design language. Clean typography, subtle borders,
# near-black accents. Refined like Stripe, Linear, Codex.

APP_STYLE = """
/* ── Foundation ─────────────────────────────────────────────────────────── */
QMainWindow {
    background: #f8f8f7;
}
QWidget {
    background: #f8f8f7;
    color: #18181b;
    font-family: "Segoe UI", "Segoe UI Variable", system-ui, sans-serif;
    font-size: 12px;
}
QLabel {
    background: transparent;
}
QWidget:disabled {
    color: #a1a1aa;
}

/* ── Top bar ────────────────────────────────────────────────────────────── */
QFrame#TopBar {
    background: #ffffff;
    border-bottom: 1px solid #e8e7e5;
}
QFrame#TopBar QLabel {
    background: transparent;
}
QLabel#AppTitle {
    color: #18181b;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: -0.3px;
}
QLabel#AppSubtitle {
    color: #6b7280;
    font-size: 11px;
    font-weight: 400;
}
QLabel#MetricLabel {
    color: #6b7280;
    font-size: 10px;
    font-weight: 500;
    background: #f4f4f5;
    border: 1px solid #e8e7e5;
    border-radius: 5px;
    padding: 4px 10px;
}

/* ── Group boxes ────────────────────────────────────────────────────────── */
QGroupBox {
    background: #ffffff;
    border: 1px solid #e8e7e5;
    border-radius: 8px;
    margin-top: 16px;
    padding: 16px 14px 14px 14px;
    font-weight: 600;
    font-size: 11px;
    color: #18181b;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    color: #6b7280;
    font-weight: 500;
    font-size: 10px;
    letter-spacing: 0.2px;
}

/* ── Tab widget ─────────────────────────────────────────────────────────── */
QTabWidget::pane {
    background: #ffffff;
    border: 1px solid #e8e7e5;
    border-radius: 0 0 8px 8px;
    top: -1px;
}
QTabBar::tab {
    background: transparent;
    color: #6b7280;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 10px 20px;
    margin-right: 0px;
    font-weight: 500;
    font-size: 12px;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #18181b;
    border-bottom: 2px solid #18181b;
    font-weight: 600;
}
QTabBar::tab:hover:!selected {
    color: #3f3f46;
    background: #f4f4f5;
    border-bottom-color: #d4d4d8;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
QPushButton {
    background: #18181b;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 7px 18px;
    min-height: 28px;
    font-weight: 600;
    font-size: 11px;
}
QPushButton:hover {
    background: #27272a;
}
QPushButton:pressed {
    background: #3f3f46;
}
QPushButton:disabled {
    background: #e8e7e5;
    color: #a1a1aa;
}
QPushButton#SecondaryButton {
    background: #ffffff;
    color: #18181b;
    border: 1px solid #d4d4d8;
}
QPushButton#SecondaryButton:hover {
    background: #f4f4f5;
    border-color: #a1a1aa;
}
QPushButton#DangerButton {
    background: #ffffff;
    color: #dc2626;
    border: 1px solid #fecaca;
}
QPushButton#DangerButton:hover {
    background: #fef2f2;
    color: #b91c1c;
    border-color: #fca5a5;
}
QPushButton#SuccessButton {
    background: #ffffff;
    color: #16a34a;
    border: 1px solid #bbf7d0;
}
QPushButton#SuccessButton:hover {
    background: #f0fdf4;
    color: #15803d;
    border-color: #86efac;
}

/* ── Inputs ─────────────────────────────────────────────────────────────── */
QLineEdit, QPlainTextEdit, QComboBox {
    background: #fafaf9;
    color: #18181b;
    border: 1px solid #d4d4d8;
    border-radius: 6px;
    padding: 8px 10px;
    selection-background-color: #e8e7e5;
    selection-color: #18181b;
}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus {
    border-color: #18181b;
}
QComboBox::drop-down {
    border: none;
    width: 28px;
    background: transparent;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #6b7280;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background: #ffffff;
    color: #18181b;
    border: 1px solid #e8e7e5;
    border-radius: 6px;
    selection-background-color: #f4f4f5;
    selection-color: #18181b;
    padding: 4px;
}

/* ── Tables ─────────────────────────────────────────────────────────────── */
QTableWidget {
    background: #ffffff;
    border: 1px solid #e8e7e5;
    border-radius: 6px;
    gridline-color: #f4f4f5;
    selection-background-color: #f4f4f5;
    selection-color: #18181b;
    alternate-background-color: #fafaf9;
    font-size: 11px;
}
QHeaderView::section {
    background: #fafaf9;
    color: #6b7280;
    border: none;
    border-bottom: 1px solid #e8e7e5;
    border-right: 1px solid #f4f4f5;
    padding: 8px 12px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.2px;
}

/* ── Progress bar ───────────────────────────────────────────────────────── */
QProgressBar {
    background: #f4f4f5;
    border: 1px solid #e8e7e5;
    border-radius: 6px;
    text-align: center;
    color: #18181b;
    font-size: 10px;
    font-weight: 600;
    height: 10px;
}
QProgressBar::chunk {
    background: #18181b;
    border-radius: 5px;
}

/* ── Splitter ───────────────────────────────────────────────────────────── */
QSplitter::handle {
    background: #e8e7e5;
    margin: 0 2px;
}
QSplitter::handle:hover {
    background: #d4d4d8;
}

/* ── Status bar ─────────────────────────────────────────────────────────── */
QStatusBar {
    background: #ffffff;
    border-top: 1px solid #e8e7e5;
    color: #6b7280;
    font-size: 10px;
    padding: 3px 12px;
}

/* ── Scroll bars ────────────────────────────────────────────────────────── */
QScrollBar:vertical {
    background: transparent;
    width: 6px;
    border-radius: 3px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #d4d4d8;
    border-radius: 3px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #a1a1aa;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background: transparent;
    height: 6px;
    border-radius: 3px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #d4d4d8;
    border-radius: 3px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background: #a1a1aa;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ── Tool tips ──────────────────────────────────────────────────────────── */
QToolTip {
    background: #18181b;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 11px;
}
"""


def compact_table(table: QTableWidget) -> None:
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QTableWidget.SelectRows)
    table.setSelectionMode(QTableWidget.SingleSelection)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(26)
    table.horizontalHeader().setStretchLastSection(True)
    table.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)


def table_item(text: object) -> QTableWidgetItem:
    cell = QTableWidgetItem("" if text is None else str(text))
    cell.setFlags(cell.flags() ^ Qt.ItemIsEditable)
    return cell


class ConfidenceBadge(QLabel):
    def set_level(self, level: str) -> None:
        styles = {
            "HIGH": ("#16a34a", "#f0fdf4", "#bbf7d0"),
            "MEDIUM": ("#ca8a04", "#fefce8", "#fde047"),
            "LOW": ("#dc2626", "#fef2f2", "#fecaca"),
        }
        fg, bg, border = styles.get(level, ("#6b7280", "#f4f4f5", "#d4d4d8"))
        self.setText(f"{level} CONFIDENCE" if level != "N/A" else "N/A")
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            f"QLabel {{ color: {fg}; background: {bg}; border: 1px solid {border}; "
            "border-radius: 4px; padding: 5px 14px; font-weight: 600; font-size: 10px; "
            "letter-spacing: 0.2px; }}"
        )


# ── Worker thread for Excel processing ─────────────────────────────────────────

class ExcelWorker(QThread):
    progress = Signal(int, int)
    finished = Signal(object, object, object)
    error = Signal(str)

    def __init__(self, file_data: bytes, filename: str, db_path: str | None = None,
                 answer_col: str = "DDQ RAG Answer", confidence_col: str = "Confidence",
                 evidence_col: str = "Source Evidence") -> None:
        super().__init__()
        self.file_data = file_data
        self.filename = filename
        self.db_path = db_path
        self.answer_col = answer_col
        self.confidence_col = confidence_col
        self.evidence_col = evidence_col

    def run(self) -> None:
        try:
            output_bytes, output_filename, summary = process_questionnaire(
                self.file_data,
                self.filename,
                db_path=self.db_path,
                progress_callback=lambda c, t: self.progress.emit(c, t),
                answer_column_name=self.answer_col,
                confidence_column_name=self.confidence_col,
                evidence_column_name=self.evidence_col,
            )
            self.finished.emit(output_bytes, output_filename, summary)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Main Window ─────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        ensure_runtime_dirs()
        self.setWindowTitle("DDQ RAG Workbench")
        self.resize(1400, 860)
        self.folder_path = Path(RAG_MATERIAL_DIR)
        self.last_answer: AnswerResponse | None = None
        self._excel_worker: ExcelWorker | None = None

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_top_bar())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_ask_tab(), "Ask Question")
        self.tabs.addTab(self._build_excel_tab(), "Process Excel")
        self.tabs.addTab(self._build_library_tab(), "Document Library")
        self.tabs.addTab(self._build_review_tab(), "Review Queue")
        self.tabs.addTab(self._build_settings_tab(), "Settings")
        root_layout.addWidget(self.tabs, 1)

        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

        self.bootstrap_default_library()
        self.refresh_all()

        # BGE-M3 is pre-loaded in main() before QApplication starts.
        # Loading torch during Qt event loop causes segfaults.
        from app.embeddings import _BGE_LOADED
        if _BGE_LOADED:
            self._set_model_ready(True)
            self.statusBar().showMessage("BGE-M3 model loaded — Ready", 5000)
        else:
            self._set_model_ready(False)
            self.statusBar().showMessage("WARNING: BGE-M3 model not loaded", 0)

    # ── Top Bar ──────────────────────────────────────────────────────────────

    def _build_top_bar(self) -> QFrame:
        top = QFrame()
        top.setObjectName("TopBar")
        layout = QHBoxLayout(top)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)

        mark = QLabel("◆")
        mark.setStyleSheet(
            "color: #18181b; font-size: 18px; font-weight: 700; "
            "background: #f4f4f5; padding: 6px 10px; border-radius: 6px;"
        )

        title_block = QVBoxLayout()
        title_block.setSpacing(1)
        title = QLabel("DDQ RAG Workbench")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Due Diligence · Compliance Intelligence")
        subtitle.setObjectName("AppSubtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        self.metric_label = QLabel("")
        self.metric_label.setObjectName("MetricLabel")
        self.metric_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(mark)
        layout.addLayout(title_block)
        layout.addStretch(1)
        layout.addWidget(self.metric_label)
        return top

    # ── Ask Tab ──────────────────────────────────────────────────────────────

    def _build_ask_tab(self) -> QWidget:
        panel = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_question_panel())
        splitter.addWidget(self._build_answer_panel())
        splitter.addWidget(self._build_evidence_panel())
        splitter.setSizes([420, 520, 440])

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        return panel

    def _build_question_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 5, 10)
        layout.setSpacing(8)

        # Source folder
        src_box = QGroupBox("Knowledge Base Source")
        src_layout = QGridLayout(src_box)
        src_layout.setContentsMargins(10, 12, 10, 10)
        src_layout.setHorizontalSpacing(6)
        src_layout.setVerticalSpacing(6)

        self.folder_input = QLineEdit(str(self.folder_path))
        browse_btn = QPushButton("Browse")
        browse_btn.setObjectName("SecondaryButton")
        browse_btn.clicked.connect(self.choose_folder)
        rebuild_btn = QPushButton("Rebuild All")
        rebuild_btn.clicked.connect(self.rebuild_folder)
        add_btn = QPushButton("Add Files")
        add_btn.setObjectName("SecondaryButton")
        add_btn.clicked.connect(self.add_files)

        src_layout.addWidget(QLabel("Folder"), 0, 0)
        src_layout.addWidget(self.folder_input, 0, 1, 1, 3)
        src_layout.addWidget(browse_btn, 0, 4)
        src_layout.addWidget(rebuild_btn, 1, 1)
        src_layout.addWidget(add_btn, 1, 2)
        layout.addWidget(src_box)

        # Question input
        q_box = QGroupBox("Inquiry")
        q_layout = QVBoxLayout(q_box)
        q_layout.setContentsMargins(10, 12, 10, 10)
        q_layout.setSpacing(8)

        self.question_input = QPlainTextEdit()
        self.question_input.setPlaceholderText(
            "Enter a due diligence question grounded in the knowledge base..."
        )
        self.question_input.setMaximumHeight(90)
        q_layout.addWidget(self.question_input)

        cmd_row = QHBoxLayout()
        self.doc_filter = QComboBox()
        self.doc_filter.addItem("All documents", "")
        ask_btn = QPushButton("Submit Inquiry")
        ask_btn.clicked.connect(self.ask_question)
        cmd_row.addWidget(QLabel("Scope"))
        cmd_row.addWidget(self.doc_filter, 1)
        cmd_row.addWidget(ask_btn)
        q_layout.addLayout(cmd_row)
        layout.addWidget(q_box)

        # Document list (compact)
        docs_box = QGroupBox("Indexed Documents")
        docs_layout = QVBoxLayout(docs_box)
        docs_layout.setContentsMargins(8, 10, 8, 8)
        self.documents_table = QTableWidget(0, 3)
        self.documents_table.setHorizontalHeaderLabels(["Document", "Chunks", "Type"])
        compact_table(self.documents_table)
        self.documents_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.documents_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.documents_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.documents_table.itemSelectionChanged.connect(self.load_selected_chunks)
        docs_layout.addWidget(self.documents_table)

        delete_btn = QPushButton("Delete Selected")
        delete_btn.setObjectName("DangerButton")
        delete_btn.clicked.connect(self.delete_selected_document)
        docs_layout.addWidget(delete_btn)

        layout.addWidget(docs_box, 1)
        return panel

    def _build_answer_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 10, 5, 10)
        layout.setSpacing(8)

        answer_box = QGroupBox("Generated Response")
        answer_layout = QVBoxLayout(answer_box)
        answer_layout.setContentsMargins(10, 12, 10, 10)
        answer_layout.setSpacing(8)

        meta_row = QHBoxLayout()
        self.confidence_badge = ConfidenceBadge("N/A")
        self.confidence_badge.set_level("N/A")
        self.similarity_label = QLabel("Top similarity: —")
        self.similarity_label.setStyleSheet(
            "color: #6b7280; font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 10px; background: transparent;"
        )
        meta_row.addWidget(self.confidence_badge)
        meta_row.addWidget(self.similarity_label)
        meta_row.addStretch(1)
        answer_layout.addLayout(meta_row)

        self.answer_text = QPlainTextEdit()
        self.answer_text.setReadOnly(True)
        self.answer_text.setPlaceholderText(
            "Formal response will appear here after retrieval clears the evidence gate..."
        )
        self.answer_text.setStyleSheet(
            "QPlainTextEdit { background: #ffffff; border: 1px solid #e8e7e5; "
            "border-left: 3px solid #18181b; border-radius: 6px; "
            "font-size: 12px; line-height: 1.6; padding: 10px; color: #18181b; }"
        )
        answer_layout.addWidget(self.answer_text, 1)
        layout.addWidget(answer_box, 1)
        return panel

    def _build_evidence_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 10, 10, 10)
        layout.setSpacing(8)

        cit_box = QGroupBox("Source Citations & Chunk Preview")
        cit_layout = QVBoxLayout(cit_box)
        cit_layout.setContentsMargins(8, 10, 8, 8)
        cit_layout.setSpacing(5)
        self.citations_table = QTableWidget(0, 4)
        self.citations_table.setHorizontalHeaderLabels(["Source", "Page", "Sim.", "Quote"])
        compact_table(self.citations_table)
        self.citations_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.citations_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.citations_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.citations_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        cit_layout.addWidget(self.citations_table)
        layout.addWidget(cit_box, 1)

        # Retrieval debug section
        debug_box = QGroupBox("Retrieval Diagnostics")
        debug_box.setMaximumHeight(140)
        debug_layout = QVBoxLayout(debug_box)
        debug_layout.setContentsMargins(8, 10, 8, 8)
        self.retrieval_debug = QPlainTextEdit()
        self.retrieval_debug.setReadOnly(True)
        self.retrieval_debug.setPlaceholderText("Retrieval metrics will appear after each query...")
        self.retrieval_debug.setStyleSheet(
            "QPlainTextEdit { background: #fafaf9; font-family: 'Cascadia Code', 'Consolas', monospace; "
            "font-size: 10px; color: #6b7280; border: 1px solid #e8e7e5; border-radius: 6px; padding: 8px; }"
        )
        debug_layout.addWidget(self.retrieval_debug)
        layout.addWidget(debug_box)

        return panel

    # ── Excel Tab ────────────────────────────────────────────────────────────

    def _build_excel_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        excel_title = QLabel("Process Excel Questionnaire")
        excel_title.setStyleSheet("color: #18181b; font-size: 16px; font-weight: 700; background: transparent;")
        excel_desc = QLabel(
            "Upload a due diligence questionnaire in Excel format. Each question will be processed "
            "through the RAG pipeline and formal answers written into new columns."
        )
        excel_desc.setStyleSheet("color: #6b7280; font-size: 11px; background: transparent;")
        excel_desc.setWordWrap(True)
        title_block.addWidget(excel_title)
        title_block.addWidget(excel_desc)
        header.addLayout(title_block, 1)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left: upload
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.setSpacing(8)

        upload_box = QGroupBox("Upload Questionnaire")
        upload_layout = QVBoxLayout(upload_box)
        upload_layout.setContentsMargins(12, 14, 12, 12)
        upload_layout.setSpacing(10)

        self.excel_path_label = QLabel("No file selected")
        self.excel_path_label.setStyleSheet(
            "color: #6b7280; background: #fafaf9; padding: 14px; border: 2px dashed "
            "#d4d4d8; border-radius: 6px; font-size: 11px;"
        )
        self.excel_path_label.setWordWrap(True)
        self.excel_path_label.setMinimumHeight(60)
        upload_layout.addWidget(self.excel_path_label)

        sel_row = QHBoxLayout()
        select_excel_btn = QPushButton("Select Excel File")
        select_excel_btn.setObjectName("SecondaryButton")
        select_excel_btn.clicked.connect(self.select_excel_file)
        sel_row.addWidget(select_excel_btn)
        sel_row.addStretch()
        upload_layout.addLayout(sel_row)

        col_opts = QGridLayout()
        col_opts.setHorizontalSpacing(8)
        col_opts.addWidget(QLabel("Answer column:"), 0, 0)
        self.excel_answer_col = QLineEdit("DDQ RAG Answer")
        col_opts.addWidget(self.excel_answer_col, 0, 1)
        col_opts.addWidget(QLabel("Confidence column:"), 1, 0)
        self.excel_confidence_col = QLineEdit("Confidence")
        col_opts.addWidget(self.excel_confidence_col, 1, 1)
        col_opts.addWidget(QLabel("Evidence column:"), 2, 0)
        self.excel_evidence_col = QLineEdit("Source Evidence")
        col_opts.addWidget(self.excel_evidence_col, 2, 1)
        upload_layout.addLayout(col_opts)

        self.start_excel_btn = QPushButton("Process Questionnaire")
        self.start_excel_btn.clicked.connect(self.process_excel)
        self.start_excel_btn.setEnabled(False)
        upload_layout.addWidget(self.start_excel_btn)

        self.excel_progress = QProgressBar()
        self.excel_progress.setVisible(False)
        upload_layout.addWidget(self.excel_progress)

        self.excel_status = QLabel("")
        self.excel_status.setStyleSheet("color: #6b7280; font-size: 10px; background: transparent;")
        self.excel_status.setWordWrap(True)
        upload_layout.addWidget(self.excel_status)

        left_layout.addWidget(upload_box)
        left_layout.addStretch()

        # Right: results
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.setSpacing(8)

        results_box = QGroupBox("Results & Download")
        results_layout = QVBoxLayout(results_box)
        results_layout.setContentsMargins(12, 14, 12, 12)
        results_layout.setSpacing(8)

        self.excel_metrics = QHBoxLayout()
        self.excel_total = QLabel("Total: —")
        self.excel_answered = QLabel("Answered: —")
        self.excel_high = QLabel("High: —")
        self.excel_errors = QLabel("Errors: —")
        for label in (self.excel_total, self.excel_answered, self.excel_high, self.excel_errors):
            label.setStyleSheet(
                "color: #18181b; font-size: 12px; font-weight: 600; "
                "background: #fafaf9; border: 1px solid #e8e7e5; padding: 10px 14px; border-radius: 6px;"
            )
            label.setAlignment(Qt.AlignCenter)
            self.excel_metrics.addWidget(label)
        results_layout.addLayout(self.excel_metrics)

        self.excel_download_btn = QPushButton("Download Answered Questionnaire")
        self.excel_download_btn.setEnabled(False)
        self.excel_download_btn.clicked.connect(self.download_excel_result)
        results_layout.addWidget(self.excel_download_btn)

        self.excel_breakdown = QTableWidget(0, 3)
        self.excel_breakdown.setHorizontalHeaderLabels(["Row", "Question", "Confidence"])
        compact_table(self.excel_breakdown)
        self.excel_breakdown.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.excel_breakdown.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.excel_breakdown.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        results_layout.addWidget(self.excel_breakdown, 1)

        right_layout.addWidget(results_box)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([420, 680])
        layout.addWidget(splitter, 1)

        return panel

    # ── Library Tab ──────────────────────────────────────────────────────────

    def _build_library_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        hl = QHBoxLayout()
        lib_title = QLabel("Document Library")
        lib_title.setStyleSheet("color: #18181b; font-size: 16px; font-weight: 700; background: transparent;")
        hl.addWidget(lib_title)
        hl.addStretch()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("SecondaryButton")
        refresh_btn.clicked.connect(self.refresh_all)
        hl.addWidget(refresh_btn)
        layout.addLayout(hl)

        self.library_table = QTableWidget(0, 6)
        self.library_table.setHorizontalHeaderLabels(
            ["Document", "Type", "Version", "Chunks", "Indexed", "State"]
        )
        compact_table(self.library_table)
        self.library_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 6):
            self.library_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.library_table.itemSelectionChanged.connect(self.load_library_chunks)
        layout.addWidget(self.library_table, 1)

        self.library_chunk_table = QTableWidget(0, 3)
        self.library_chunk_table.setHorizontalHeaderLabels(["Source", "Page", "Text Preview"])
        compact_table(self.library_chunk_table)
        self.library_chunk_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.library_chunk_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.library_chunk_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        layout.addWidget(self.library_chunk_table, 1)

        return panel

    # ── Review Tab ───────────────────────────────────────────────────────────

    def _build_review_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        hl = QHBoxLayout()
        review_title = QLabel("Human Review Queue")
        review_title.setStyleSheet("color: #18181b; font-size: 16px; font-weight: 700; background: transparent;")
        hl.addWidget(review_title)
        hl.addStretch()

        for st, label in [("PENDING", "Pending"), ("APPROVED", "Approved"), ("REJECTED", "Rejected")]:
            btn = QPushButton(label)
            btn.setObjectName("SecondaryButton")
            btn.clicked.connect(lambda checked, s=st: self.set_review_filter(s))
            hl.addWidget(btn)

        hl.addWidget(QPushButton("All", objectName="SecondaryButton",
                                 clicked=lambda: self.set_review_filter(None)))
        hl.addStretch(1)
        clear_all_btn = QPushButton("Clear All History")
        clear_all_btn.setObjectName("DangerButton")
        clear_all_btn.clicked.connect(self.clear_all_reviews)
        hl.addWidget(clear_all_btn)
        layout.addLayout(hl)

        self.review_table = QTableWidget(0, 5)
        self.review_table.setHorizontalHeaderLabels(["Status", "Reason", "Question", "Answer Preview", "Created"])
        compact_table(self.review_table)
        self.review_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.review_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.review_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.review_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.review_table.itemSelectionChanged.connect(self.load_review_detail)
        layout.addWidget(self.review_table, 1)

        btn_row = QHBoxLayout()
        approve_btn = QPushButton("Approve Selected")
        approve_btn.setObjectName("SecondaryButton")
        approve_btn.clicked.connect(lambda: self.update_selected_review("APPROVED"))
        reject_btn = QPushButton("Reject Selected")
        reject_btn.setObjectName("DangerButton")
        reject_btn.clicked.connect(lambda: self.update_selected_review("REJECTED"))
        btn_row.addWidget(approve_btn)
        btn_row.addWidget(reject_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.review_detail = QPlainTextEdit()
        self.review_detail.setReadOnly(True)
        self.review_detail.setMaximumHeight(120)
        self.review_detail.setPlaceholderText("Select a review item to see full answer text...")
        layout.addWidget(self.review_detail)

        return panel

    # ── Settings Tab ──────────────────────────────────────────────────────────

    def _build_settings_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        hl = QHBoxLayout()
        settings_title = QLabel("Configuration")
        settings_title.setStyleSheet("color: #18181b; font-size: 16px; font-weight: 700; background: transparent;")
        hl.addWidget(settings_title)
        hl.addStretch()
        layout.addLayout(hl)

        # Use a scroll area for smaller windows
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)

        # ── Generation Provider ──
        gen_box = QGroupBox("Generation Provider")
        gen_layout = QVBoxLayout(gen_box)
        gen_layout.setContentsMargins(14, 16, 14, 14)
        gen_layout.setSpacing(10)

        prov_row = QHBoxLayout()
        prov_row.addWidget(QLabel("Provider:"))
        self.settings_provider = QComboBox()
        self.settings_provider.addItem("Google Gemini", "gemini")
        self.settings_provider.addItem("OpenRouter", "openrouter")
        self.settings_provider.addItem("Local (Extractive — no LLM)", "local")
        self.settings_provider.currentIndexChanged.connect(self._on_provider_changed)
        prov_row.addWidget(self.settings_provider, 1)
        gen_layout.addLayout(prov_row)
        content_layout.addWidget(gen_box)

        # ── Gemini Settings ──
        self.gemini_box = QGroupBox("Google Gemini API")
        gemini_layout = QVBoxLayout(self.gemini_box)
        gemini_layout.setContentsMargins(14, 16, 14, 14)
        gemini_layout.setSpacing(8)

        gkey_row = QHBoxLayout()
        gkey_row.addWidget(QLabel("API Key:"))
        self.settings_gemini_key = QLineEdit()
        self.settings_gemini_key.setEchoMode(QLineEdit.Password)
        self.settings_gemini_key.setPlaceholderText("Enter Gemini API key...")
        gkey_row.addWidget(self.settings_gemini_key, 1)
        show_gkey = QPushButton("Show")
        show_gkey.setObjectName("SecondaryButton")
        show_gkey.setFixedWidth(60)
        show_gkey.clicked.connect(lambda: self._toggle_password(self.settings_gemini_key))
        gkey_row.addWidget(show_gkey)
        gemini_layout.addLayout(gkey_row)

        gmodel_row = QHBoxLayout()
        gmodel_row.addWidget(QLabel("Model:"))
        self.settings_gemini_model = QLineEdit()
        self.settings_gemini_model.setPlaceholderText("gemini-2.5-flash")
        gmodel_row.addWidget(self.settings_gemini_model, 1)
        gemini_layout.addLayout(gmodel_row)
        content_layout.addWidget(self.gemini_box)

        # ── OpenRouter Settings ──
        self.openrouter_box = QGroupBox("OpenRouter API")
        or_layout = QVBoxLayout(self.openrouter_box)
        or_layout.setContentsMargins(14, 16, 14, 14)
        or_layout.setSpacing(8)

        orkey_row = QHBoxLayout()
        orkey_row.addWidget(QLabel("API Key:"))
        self.settings_openrouter_key = QLineEdit()
        self.settings_openrouter_key.setEchoMode(QLineEdit.Password)
        self.settings_openrouter_key.setPlaceholderText("Enter OpenRouter API key...")
        orkey_row.addWidget(self.settings_openrouter_key, 1)
        show_orkey = QPushButton("Show")
        show_orkey.setObjectName("SecondaryButton")
        show_orkey.setFixedWidth(60)
        show_orkey.clicked.connect(lambda: self._toggle_password(self.settings_openrouter_key))
        orkey_row.addWidget(show_orkey)
        or_layout.addLayout(orkey_row)

        ormodel_row = QHBoxLayout()
        ormodel_row.addWidget(QLabel("Model:"))
        self.settings_openrouter_model = QLineEdit()
        self.settings_openrouter_model.setPlaceholderText("anthropic/claude-sonnet-4-6")
        ormodel_row.addWidget(self.settings_openrouter_model, 1)
        or_layout.addLayout(ormodel_row)
        content_layout.addWidget(self.openrouter_box)

        # ── Embedding Model ──
        emb_box = QGroupBox("Embedding Model (BGE-M3)")
        emb_layout = QVBoxLayout(emb_box)
        emb_layout.setContentsMargins(14, 16, 14, 14)
        emb_layout.setSpacing(8)

        bge_row = QHBoxLayout()
        bge_row.addWidget(QLabel("Model Path:"))
        self.settings_bge_path = QLineEdit()
        self.settings_bge_path.setPlaceholderText("Absolute path to BGE-M3 model directory...")
        bge_row.addWidget(self.settings_bge_path, 1)
        browse_bge = QPushButton("Browse")
        browse_bge.setObjectName("SecondaryButton")
        browse_bge.clicked.connect(self._browse_bge_path)
        bge_row.addWidget(browse_bge)
        emb_layout.addLayout(bge_row)
        content_layout.addWidget(emb_box)

        # ── Storage ──
        db_box = QGroupBox("Storage")
        db_layout = QVBoxLayout(db_box)
        db_layout.setContentsMargins(14, 16, 14, 14)
        db_layout.setSpacing(8)

        db_row = QHBoxLayout()
        db_row.addWidget(QLabel("Database:"))
        self.settings_db_path = QLineEdit()
        self.settings_db_path.setPlaceholderText("Path to SQLite database...")
        db_row.addWidget(self.settings_db_path, 1)
        browse_db = QPushButton("Browse")
        browse_db.setObjectName("SecondaryButton")
        browse_db.clicked.connect(self._browse_db_path)
        db_row.addWidget(browse_db)
        db_layout.addLayout(db_row)
        content_layout.addWidget(db_box)

        layout.addWidget(content, 1)

        # ── Actions ──
        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save_settings)
        reset_btn = QPushButton("Reload from .env")
        reset_btn.setObjectName("SecondaryButton")
        reset_btn.clicked.connect(self._load_settings)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()

        self.settings_status = QLabel("")
        self.settings_status.setStyleSheet("color: #6b7280; font-size: 11px; background: transparent;")
        btn_row.addWidget(self.settings_status)
        layout.addLayout(btn_row)

        self._load_settings()
        return panel

    def _toggle_password(self, field: QLineEdit) -> None:
        if field.echoMode() == QLineEdit.Password:
            field.setEchoMode(QLineEdit.Normal)
        else:
            field.setEchoMode(QLineEdit.Password)

    def _browse_bge_path(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select BGE-M3 Model Directory")
        if path:
            self.settings_bge_path.setText(path)

    def _browse_db_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Select Database File", "", "SQLite (*.sqlite3 *.db *.sqlite)")
        if path:
            self.settings_db_path.setText(path)

    def _load_settings(self) -> None:
        import os as _os
        provider = _os.getenv("DDQ_GENERATION_PROVIDER", "gemini").lower()
        idx = self.settings_provider.findData(provider)
        if idx >= 0:
            self.settings_provider.setCurrentIndex(idx)
        self.settings_gemini_key.setText(_os.getenv("GEMINI_API_KEY", ""))
        self.settings_gemini_model.setText(_os.getenv("GEMINI_MODEL", ""))
        self.settings_openrouter_key.setText(_os.getenv("OPENROUTER_API_KEY", ""))
        self.settings_openrouter_model.setText(_os.getenv("OPENROUTER_LLM_MODEL", ""))
        self.settings_bge_path.setText(_os.getenv("BGE_MODEL_PATH", ""))
        db_path = _os.getenv("DDQ_DB_PATH", "")
        self.settings_db_path.setText(db_path)
        self._on_provider_changed()

    def _on_provider_changed(self) -> None:
        provider = self.settings_provider.currentData()
        self.gemini_box.setVisible(provider == "gemini")
        self.openrouter_box.setVisible(provider == "openrouter")

    def _save_settings(self) -> None:
        from pathlib import Path as _Path
        import sys as _sys

        # When frozen (PyInstaller), write next to the .exe; otherwise use project root
        if getattr(_sys, "frozen", False):
            env_path = _Path(_sys.executable).parent / ".env"
        else:
            env_path = _Path(__file__).resolve().parents[1] / ".env"

        env_vars = {}
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_stripped = line.strip()
                    if not line_stripped or line_stripped.startswith("#") or "=" not in line_stripped:
                        continue
                    key, _, value = line_stripped.partition("=")
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")

        provider = self.settings_provider.currentData()
        env_vars["DDQ_GENERATION_PROVIDER"] = provider
        env_vars["GEMINI_API_KEY"] = self.settings_gemini_key.text().strip()
        env_vars["GEMINI_MODEL"] = self.settings_gemini_model.text().strip()
        env_vars["OPENROUTER_API_KEY"] = self.settings_openrouter_key.text().strip()
        env_vars["OPENROUTER_LLM_MODEL"] = self.settings_openrouter_model.text().strip()
        env_vars["BGE_MODEL_PATH"] = self.settings_bge_path.text().strip()
        db_val = self.settings_db_path.text().strip()
        if db_val:
            env_vars["DDQ_DB_PATH"] = db_val

        env_vars = {k: v for k, v in env_vars.items() if v}

        lines = []
        lines.append("DDQ_DB_PATH=" + env_vars.pop("DDQ_DB_PATH", "data/ddqrag.sqlite3"))
        lines.append("DDQ_GENERATION_PROVIDER=" + env_vars.pop("DDQ_GENERATION_PROVIDER", "gemini"))
        lines.append("DDQ_EMBEDDING_PROVIDER=" + env_vars.pop("DDQ_EMBEDDING_PROVIDER", "local"))
        lines.append("")
        if env_vars.get("GEMINI_API_KEY"):
            lines.append("# Google Gemini for answer generation")
            lines.append("GEMINI_API_KEY=" + env_vars.pop("GEMINI_API_KEY"))
            if "GEMINI_MODEL" in env_vars:
                lines.append("GEMINI_MODEL=" + env_vars.pop("GEMINI_MODEL"))
            lines.append("")
        if env_vars.get("OPENROUTER_API_KEY"):
            lines.append("# OpenRouter (alternative LLM provider)")
            lines.append("OPENROUTER_API_KEY=" + env_vars.pop("OPENROUTER_API_KEY"))
            if "OPENROUTER_LLM_MODEL" in env_vars:
                lines.append("OPENROUTER_LLM_MODEL=" + env_vars.pop("OPENROUTER_LLM_MODEL"))
            lines.append("")
        if env_vars.get("BGE_MODEL_PATH"):
            lines.append("# Local BGE-M3 model — absolute path to downloaded model directory (REQUIRED)")
            lines.append("BGE_MODEL_PATH=" + env_vars.pop("BGE_MODEL_PATH"))
            lines.append("")
        for k, v in env_vars.items():
            lines.append(f"{k}={v}")

        with open(env_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        # Update runtime environment
        import os as _os
        _os.environ["DDQ_GENERATION_PROVIDER"] = provider
        _os.environ["GEMINI_API_KEY"] = self.settings_gemini_key.text().strip()
        _os.environ["GEMINI_MODEL"] = self.settings_gemini_model.text().strip()
        _os.environ["OPENROUTER_API_KEY"] = self.settings_openrouter_key.text().strip()
        _os.environ["OPENROUTER_LLM_MODEL"] = self.settings_openrouter_model.text().strip()
        _os.environ["BGE_MODEL_PATH"] = self.settings_bge_path.text().strip()

        self.settings_status.setText("Settings saved — restart recommended for full effect")
        self.settings_status.setStyleSheet("color: #16a34a; font-size: 11px; font-weight: 500; background: transparent;")
        self.statusBar().showMessage("Settings saved to .env", 5000)

    # ── Logic: Library ───────────────────────────────────────────────────────

    def _set_model_ready(self, ready: bool) -> None:
        """Enable or disable controls that require the embedding model."""
        for btn in self.findChildren(QPushButton):
            if btn.text() in ("Submit Inquiry", "Process Questionnaire", "Rebuild All", "Add Files"):
                btn.setEnabled(ready)

    def bootstrap_default_library(self) -> None:
        if list_documents():
            return
        if not self.folder_path.exists():
            return
        try:
            index_folder(self.folder_path)
        except Exception:
            return

    def refresh_all(self) -> None:
        self.refresh_documents()
        self.refresh_review_queue()
        self.refresh_library_tab()
        self.statusBar().showMessage("Ready")

    def refresh_documents(self) -> None:
        rows = list_documents()
        stale = {row["doc_name"]: row.get("stale_reason", "stale") for row in stale_documents()}
        self.documents_table.setRowCount(len(rows))
        self.doc_filter.blockSignals(True)
        self.doc_filter.clear()
        self.doc_filter.addItem("All documents", "")
        for row_index, row in enumerate(rows):
            state = stale.get(row["doc_name"], "current")
            self.documents_table.setItem(row_index, 0, table_item(row["doc_name"]))
            self.documents_table.setItem(row_index, 1, table_item(row.get("chunk_count", "")))
            self.documents_table.setItem(row_index, 2, table_item(row.get("doc_type", "")))
            self.doc_filter.addItem(row["doc_name"], row["doc_name"])
        self.doc_filter.blockSignals(False)
        total_chunks = sum(int(row.get("chunk_count") or 0) for row in rows)
        pending = len(list_review_items(status="PENDING"))
        self.metric_label.setText(
            f"DOCUMENTS {len(rows)}  |  CHUNKS {total_chunks}  |  PENDING REVIEW {pending}"
        )

    def refresh_review_queue(self) -> None:
        rows = list_review_items(status=self._review_filter if hasattr(self, "_review_filter") else None)
        self.review_table.setRowCount(len(rows))
        for row_index, review in enumerate(rows):
            self.review_table.setItem(row_index, 0, table_item(review.status))
            self.review_table.setItem(row_index, 1, table_item(review.reason))
            self.review_table.setItem(row_index, 2, table_item(review.question))
            answer = review.answer_json.get("answer", "")[:150] if isinstance(review.answer_json, dict) else ""
            self.review_table.setItem(row_index, 3, table_item(answer))
            self.review_table.setItem(row_index, 4, table_item(review.created_at[:19].replace("T", " ")))
            self.review_table.item(row_index, 0).setData(Qt.UserRole, review.item_id)
        self.refresh_documents()

    def refresh_library_tab(self) -> None:
        rows = list_documents()
        stale = {row["doc_name"]: row.get("stale_reason", "stale") for row in stale_documents()}
        self.library_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            state = stale.get(row["doc_name"], "current")
            self.library_table.setItem(row_index, 0, table_item(row["doc_name"]))
            self.library_table.setItem(row_index, 1, table_item(row.get("doc_type", "")))
            self.library_table.setItem(row_index, 2, table_item(row.get("version", "")))
            self.library_table.setItem(row_index, 3, table_item(row.get("chunk_count", "")))
            self.library_table.setItem(row_index, 4, table_item(row.get("indexed_at", "")[:19].replace("T", " ")))
            self.library_table.setItem(row_index, 5, table_item(state))

    def set_review_filter(self, status: str | None) -> None:
        self._review_filter = status
        self.refresh_review_queue()

    def load_review_detail(self) -> None:
        selected = self.review_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        answer = self.review_table.item(row, 3).text() if self.review_table.item(row, 3) else ""
        self.review_detail.setPlainText(answer)

    # ── Logic: Folder & Documents ────────────────────────────────────────────

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select RAG Material Folder", str(self.folder_path))
        if folder:
            self.folder_path = Path(folder)
            self.folder_input.setText(folder)

    def rebuild_folder(self) -> None:
        folder = Path(self.folder_input.text().strip())
        if not folder.exists():
            QMessageBox.warning(self, "Folder not found", "The selected folder does not exist.")
            return
        self.statusBar().showMessage("Indexing folder...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            results = index_folder(folder)
        except Exception as exc:
            QMessageBox.critical(self, "Index Failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.refresh_all()
        self.statusBar().showMessage(f"Indexed {len(results)} documents from {folder}", 7000)

    def add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add Documents", str(self.folder_path),
            "Documents (*.pdf *.docx *.txt *.md)",
        )
        if not files:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for path in files:
                index_document(path)
        except Exception as exc:
            QMessageBox.critical(self, "Add Files Failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.refresh_all()
        self.statusBar().showMessage(f"Indexed {len(files)} files", 7000)

    def selected_doc_name(self) -> str | None:
        selected = self.documents_table.selectedItems()
        if not selected:
            return None
        row = selected[0].row()
        item = self.documents_table.item(row, 0)
        return item.text() if item else None

    def delete_selected_document(self) -> None:
        doc_name = self.selected_doc_name()
        if not doc_name:
            return
        choice = QMessageBox.question(
            self, "Delete from Library",
            f"Remove all indexed chunks for '{doc_name}'?\nThis cannot be undone.",
        )
        if choice != QMessageBox.StandardButton.Yes:
            return
        delete_library_document(doc_name)
        self.refresh_all()
        self.citations_table.setRowCount(0)

    def load_selected_chunks(self) -> None:
        doc_name = self.selected_doc_name()
        if not doc_name:
            return
        chunks = list_chunks(doc_name=doc_name)
        self.citations_table.setRowCount(len(chunks))
        for row_index, chunk in enumerate(chunks):
            self.citations_table.setItem(row_index, 0, table_item(chunk.doc_name))
            self.citations_table.setItem(row_index, 1, table_item(chunk.page))
            self.citations_table.setItem(row_index, 2, table_item(chunk.text[:260]))

    def load_library_chunks(self) -> None:
        selected = self.library_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        item = self.library_table.item(row, 0)
        if not item:
            return
        doc_name = item.text()
        chunks = list_chunks(doc_name=doc_name)
        self.library_chunk_table.setRowCount(len(chunks))
        for row_index, chunk in enumerate(chunks):
            self.library_chunk_table.setItem(row_index, 0, table_item(chunk.doc_name))
            self.library_chunk_table.setItem(row_index, 1, table_item(chunk.page))
            self.library_chunk_table.setItem(row_index, 2, table_item(chunk.text[:260]))

    def clear_all_reviews(self) -> None:
        count = sum(1 for _ in list_review_items())
        if count == 0:
            self.statusBar().showMessage("Review queue is already empty", 4000)
            return
        choice = QMessageBox.question(
            self, "Clear Review History",
            f"Permanently delete all {count} review item(s)?\nThis cannot be undone.",
        )
        if choice != QMessageBox.StandardButton.Yes:
            return
        clear_all_reviews()
        self.refresh_review_queue()
        self.statusBar().showMessage(f"Cleared {count} review item(s)", 5000)

    def update_selected_review(self, status: str) -> None:
        selected = self.review_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        item = self.review_table.item(row, 0)
        if not item:
            return
        item_id = item.data(Qt.UserRole)
        if not item_id:
            return
        update_review_status(item_id, status)
        self.refresh_review_queue()
        self.statusBar().showMessage(f"Review item marked {status}", 5000)

    # ── Logic: Ask ───────────────────────────────────────────────────────────

    def ask_question(self) -> None:
        question = self.question_input.toPlainText().strip()
        if not question:
            return
        doc_name = self.doc_filter.currentData() or None
        self.statusBar().showMessage("Retrieving evidence...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = run_question(question, doc_name=doc_name)
        except Exception as exc:
            QMessageBox.critical(self, "Inquiry Failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.last_answer = result.response
        self.render_answer(result.response, result.top_similarity)
        self.render_citations(result.response)
        self.render_retrieval_debug(result)
        self.refresh_documents()
        if result.review_required:
            self.statusBar().showMessage(f"Escalated to review — {result.review_reason}", 9000)
        else:
            self.statusBar().showMessage("Answer generated with evidence citations", 7000)

    def render_answer(self, response: AnswerResponse, top_similarity: float) -> None:
        self.confidence_badge.set_level(response.confidence_level)
        self.similarity_label.setText(f"Top similarity: {top_similarity:.4f}")
        lines = [response.answer]
        if response.uncertainty_note:
            lines.append("")
            lines.append(f"Limitation: {response.uncertainty_note}")
        self.answer_text.setPlainText("\n".join(lines))

    def render_citations(self, response: AnswerResponse) -> None:
        self.citations_table.setRowCount(len(response.source_citations))
        for row_index, citation in enumerate(response.source_citations):
            self.citations_table.setItem(row_index, 0, table_item(citation.doc_name))
            self.citations_table.setItem(row_index, 1, table_item(citation.page))
            self.citations_table.setItem(row_index, 2, table_item(getattr(citation, 'similarity', '—')))
            self.citations_table.setItem(row_index, 3, table_item(citation.quote))

    def render_retrieval_debug(self, result: PipelineResult) -> None:
        summary = result.retrieval_summary
        chunks = result.retrieved_chunks
        if not summary and not chunks:
            self.retrieval_debug.setPlainText("")
            return

        lines = []
        if summary:
            lines.append(f"DB chunks: {summary.total_chunks_searched} | "
                         f"Passed threshold ({summary.threshold:.2f}): {summary.chunks_passed_threshold} | "
                         f"Top score: {summary.top_score:.4f}")

        if chunks:
            lines.append("—" * 40)
            for i, chunk in enumerate(chunks[:10]):
                marker = "+" if chunk.similarity >= 0.45 else "-"
                lines.append(
                    f"{marker} [{chunk.similarity:.4f}] {chunk.doc_name}:{chunk.page}  "
                    f"{chunk.text[:80].replace(chr(10), ' ')}"
                )

        self.retrieval_debug.setPlainText("\n".join(lines))

    # ── Logic: Excel ─────────────────────────────────────────────────────────

    def select_excel_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Excel Questionnaire", "",
            "Excel Files (*.xlsx *.xls)",
        )
        if file_path:
            self._excel_file_path = file_path
            self.excel_path_label.setText(f"Selected: {Path(file_path).name}")
            self.excel_path_label.setStyleSheet(
                "color: #18181b; background: #f4f4f5; padding: 14px; border: 1px solid "
                "#d4d4d8; border-radius: 6px; font-size: 11px; font-weight: 600;"
            )
            self.start_excel_btn.setEnabled(True)

    def process_excel(self) -> None:
        if not hasattr(self, "_excel_file_path"):
            return
        file_path = self._excel_file_path
        try:
            with open(file_path, "rb") as f:
                file_data = f.read()
        except Exception as exc:
            QMessageBox.critical(self, "Read Error", str(exc))
            return

        self.start_excel_btn.setEnabled(False)
        self.excel_progress.setVisible(True)
        self.excel_progress.setValue(0)
        self.excel_status.setText("Processing...")

        self._excel_worker = ExcelWorker(
            file_data,
            Path(file_path).name,
            answer_col=self.excel_answer_col.text().strip() or "DDQ RAG Answer",
            confidence_col=self.excel_confidence_col.text().strip() or "Confidence",
            evidence_col=self.excel_evidence_col.text().strip() or "Source Evidence",
        )
        self._excel_worker.progress.connect(self._on_excel_progress)
        self._excel_worker.finished.connect(self._on_excel_finished)
        self._excel_worker.error.connect(self._on_excel_error)
        self._excel_worker.start()

    def _on_excel_progress(self, current: int, total: int) -> None:
        self.excel_progress.setMaximum(total)
        self.excel_progress.setValue(current)
        self.excel_status.setText(f"Processing question {current} of {total}...")

    def _on_excel_finished(self, output_bytes: bytes, filename: str, summary: dict) -> None:
        self._excel_output = output_bytes
        self._excel_output_name = filename
        self._excel_summary = summary

        self.excel_progress.setVisible(False)
        self.start_excel_btn.setEnabled(True)

        self.excel_total.setText(f"Total: {summary['total']}")
        self.excel_answered.setText(f"Answered: {summary['answered']}")
        self.excel_high.setText(f"High: {summary['high_confidence']}")
        self.excel_errors.setText(f"Errors: {summary['errors']}")

        self.excel_download_btn.setEnabled(True)
        self.excel_status.setText(
            f"Complete — {summary['answered']}/{summary['total']} answered "
            f"({summary['high_confidence']} high confidence, {summary['errors']} errors)"
        )

        # Breakdown table
        questions = summary.get("questions", [])
        self.excel_breakdown.setRowCount(len(questions))
        for i, q in enumerate(questions):
            self.excel_breakdown.setItem(i, 0, table_item(q["row"]))
            self.excel_breakdown.setItem(i, 1, table_item(q["question"]))
            self.excel_breakdown.setItem(i, 2, table_item(q["confidence"]))

        self.refresh_documents()

    def _on_excel_error(self, error_msg: str) -> None:
        self.excel_progress.setVisible(False)
        self.start_excel_btn.setEnabled(True)
        self.excel_status.setText(f"Error: {error_msg}")
        QMessageBox.critical(self, "Processing Error", error_msg)

    def download_excel_result(self) -> None:
        if not hasattr(self, "_excel_output"):
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Answered Questionnaire",
            self._excel_output_name,
            "Excel Files (*.xlsx)",
        )
        if save_path:
            Path(save_path).write_bytes(self._excel_output)
            self.statusBar().showMessage(f"Saved to {save_path}", 7000)


# ── Entry Point ─────────────────────────────────────────────────────────────────

def main() -> None:
    # Load BGE-M3 before Qt — torch init during Qt event loop causes segfaults.
    # This blocks for 30-90s but guarantees a stable process.
    try:
        import app.embeddings as _emb
        _emb._load_bge_model()
        if _emb._BGE_MODEL is None:
            raise RuntimeError("BGE-M3 model returned None after loading")
    except Exception as exc:
        # Can't show Qt dialog yet — exit cleanly (no console in windowed build)
        raise SystemExit(1) from exc

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    window = MainWindow()
    window.show()

    if "--smoke-test" in sys.argv:
        if window.documents_table.rowCount() < 1:
            raise SystemExit(2)
        window.close()
        raise SystemExit(0)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()