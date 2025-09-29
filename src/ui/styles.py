from qtpy import QtCore

STYLE_SHEET = """
#Root { background: #0e1116; }

/* Hero */
#Hero { 
    background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
        stop:0 #1b2330, stop:1 #151a22);
    border: 1px solid #1f2835;
    border-radius: 16px;
}

QLabel#H1 {
    color: #e6edf3;
    font-size: 22px;
    font-weight: 700;
}
QLabel#SubH1 {
    color: #9fb0c1;
    font-size: 13px;
}

QLabel#H2 {
    color: #cbd5e1;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.2px;
}

QLabel#FieldLabel {
    color: #9fb0c1;
    font-size: 12px;
}

/* General text fields */
QLineEdit {
    background: #0b0f14;
    color: #d6e2f0;
    border: 1px solid #243042;
    border-radius: 12px;
    padding: 10px 12px;
    selection-background-color: #385375;
}
QLineEdit:focus {
    border: 1px solid #3a84ff;
}
QLineEdit[placeholderText] { color: #6b7d91; }

/* Buttons */
QPushButton { 
    border-radius: 12px; 
    padding: 10px 14px; 
    font-weight: 600; 
}
QPushButton#PrimaryBtn { 
    background: #1f7aec; color: white; border: 1px solid #1b65c2; 
}
QPushButton#PrimaryBtn:hover { background: #2a86ff; }

QPushButton#SecondaryBtn {
    background: #141a23; color: #cbd5e1; border: 1px solid #2b3a4d;
}
QPushButton#SecondaryBtn:hover { border-color: #3a84ff; color: #e6edf3; }

QPushButton#GhostBtn { 
    background: transparent; color: #93a6bb; border: 1px solid #2b3a4d; 
}
QPushButton#GhostBtn:hover { color: #cbd5e1; border-color: #3a84ff; }

QPushButton#CTA { 
    background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
        stop:0 #2b9bff, stop:1 #6a5cff);
    color: white; border: none; 
    padding: 12px 18px; font-size: 14px; 
}
QPushButton#CTA:disabled { background: #1a2230; color: #5f728a; }
QPushButton#CTA:hover:!disabled { filter: brightness(1.05); }

/* Cards */
#Card, #SelectableCard {
    background: #0b0f14;
    border: 1px solid #1c2634;
    border-radius: 16px;
}
#Card { box-shadow: 0 8px 24px rgba(0,0,0,0.28); }

/* SelectableCard states */
#SelectableCard[active="true"] {
    border: 1px solid #3a84ff;
    box-shadow: 0 0 0 3px rgba(58,132,255,0.18);
}

QLabel#CardTitle { color: #e6edf3; font-weight: 600; }
QLabel#CardSubtitle { color: #9fb0c1; font-size: 12px; }


#ModernTabs::pane {
    border: none;
    padding-top: 4px;
}

#ModernTabs QTabBar {
    qproperty-drawBase: 0;
    /* Slightly smaller default for better fit; tweak as you like */
    font-size: 12px;
    /* Let the bar decide widths naturally; scroll buttons handle overflow */
}

#ModernTabs QTabBar::tab {
    background: #0b0f14;
    color: #b8c5d4;
    border: 1px solid #1c2634;
    border-radius: 12px;
    padding: 8px 16px;            /* a bit more horizontal padding helps legibility */
    margin-right: 8px;
    margin-top: 6px;
    font-weight: 600;
    letter-spacing: 0.1px;        /* reduced spacing to fit more text */
    /* No fixed width here, so the tab grows to content.
       QTabWidget will show scroll buttons rather than ellipsize. */
}

#ModernTabs QTabBar::tab:selected {
    background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
        stop:0 #2b9bff, stop:1 #6a5cff);
    color: white;
    border: 1px solid #2f6ee9;
}

#ModernTabs QTabBar::tab:!selected:hover {
    border-color: #3a84ff;
    color: #dbe6f3;
}

#ModernTabs QTabBar::tab:disabled {
    color: #6b7d91;
    border-color: #1a2330;
}

#ModernTabs QTabBar::scroller { width: 18px; }
#ModernTabs QToolButton { border: none; }

"""

# Optional small helper for spacing policies across the app
DEFAULT_CONTENT_MARGINS = (12, 12, 12, 12)
DEFAULT_SPACING = 12
