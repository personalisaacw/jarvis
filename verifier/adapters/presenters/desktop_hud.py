import threading
import json
from typing import Optional, Callable
from ...ports.presenter import IReviewPresenterPort
from ...domain.entities import ReviewSession, DiffHunk


class DesktopHUDPresenter(IReviewPresenterPort):
    """
    Floating TopMost desktop HUD overlay for sequential hunk review.
    
    Uses CustomTkinter for a sleek, semi-translucent, borderless window
    that hovers over VS Code / Cursor / terminal during code review.
    Supports keyboard hotkeys: [Y] Accept, [N] Reject, [A] Accept All,
    [R] Reject All, [E] Explain, [Space] Skip.
    """

    def __init__(self, on_command: Optional[Callable[[str, Optional[str]], None]] = None):
        """
        Args:
            on_command: Callback fired when user interacts via HUD.
                        Signature: (action: str, hunk_id: Optional[str]) -> None
                        Actions: 'accept', 'reject', 'accept_all', 'reject_all', 'explain', 'skip'
        """
        self.on_command = on_command
        self._root = None
        self._diff_label = None
        self._header_label = None
        self._progress_label = None
        self._status_label = None
        self._current_hunk_id: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._is_open = False

    def on_session_started(self, session: ReviewSession) -> None:
        """Open the floating HUD window when a review session begins."""
        total = len(session.all_hunks)
        file_count = len(session.file_diffs)
        print(f"[DesktopHUD] Review session started: {file_count} file(s), {total} hunk(s)")
        self._open_window()

    def on_hunk_displayed(self, session: ReviewSession, hunk: DiffHunk) -> None:
        """Update the HUD to show the current hunk."""
        self._current_hunk_id = hunk.hunk_id
        total = len(session.all_hunks)
        idx = session.current_hunk_index + 1

        if self._root and self._is_open:
            try:
                self._root.after(0, self._update_display, hunk, idx, total)
            except Exception:
                pass

    def on_hunk_processed(self, session: ReviewSession, hunk: DiffHunk) -> None:
        """Flash a brief status indicator after a hunk decision."""
        status_text = f"✓ Block {hunk.status.value}" if hunk.status.value == "accepted" else f"✗ Block {hunk.status.value}"
        if self._root and self._is_open:
            try:
                self._root.after(0, self._flash_status, status_text)
            except Exception:
                pass

    def on_session_completed(self, session: ReviewSession) -> None:
        """Close the HUD window when the review session ends."""
        accepted = session.accepted_count
        rejected = session.rejected_count
        print(f"[DesktopHUD] Session complete — {accepted} accepted, {rejected} rejected")
        self._close_window()

    def on_explanation_ready(self, hunk: DiffHunk, explanation: str) -> None:
        """Display an explanation tooltip on the HUD."""
        if self._root and self._is_open:
            try:
                self._root.after(0, self._show_explanation, explanation)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def _open_window(self) -> None:
        """Launch the HUD in a background thread so it doesn't block the main loop."""
        if self._is_open:
            return
        self._thread = threading.Thread(target=self._create_window, daemon=True)
        self._thread.start()

    def _close_window(self) -> None:
        """Gracefully close the HUD window."""
        self._is_open = False
        if self._root:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass
        self._root = None

    def _create_window(self) -> None:
        """Build the floating HUD using tkinter (standard library)."""
        try:
            import tkinter as tk
            from tkinter import font as tkfont
        except ImportError:
            print("[DesktopHUD] tkinter not available — HUD will not display")
            return

        self._root = tk.Tk()
        root = self._root
        self._is_open = True

        # ── Window configuration ──
        root.title("JARVIS Code Review")
        root.overrideredirect(True)          # Frameless
        root.attributes("-topmost", True)    # Always-on-top
        root.attributes("-alpha", 0.93)      # Slight translucency
        root.configure(bg="#1e1e1e")

        # Position: bottom-right of primary monitor
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        win_w, win_h = 520, 420
        x = screen_w - win_w - 30
        y = screen_h - win_h - 80
        root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        # ── Drag support (since window is frameless) ──
        self._drag_data = {"x": 0, "y": 0}

        def start_drag(event):
            self._drag_data["x"] = event.x
            self._drag_data["y"] = event.y

        def do_drag(event):
            dx = event.x - self._drag_data["x"]
            dy = event.y - self._drag_data["y"]
            new_x = root.winfo_x() + dx
            new_y = root.winfo_y() + dy
            root.geometry(f"+{new_x}+{new_y}")

        # ── Header bar (draggable) ──
        header_frame = tk.Frame(root, bg="#007acc", height=36)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        header_frame.bind("<Button-1>", start_drag)
        header_frame.bind("<B1-Motion>", do_drag)

        self._header_label = tk.Label(
            header_frame, text="🧠 JARVIS Code Review",
            bg="#007acc", fg="white",
            font=("Segoe UI", 11, "bold"), anchor="w", padx=10
        )
        self._header_label.pack(side="left", fill="y")
        self._header_label.bind("<Button-1>", start_drag)
        self._header_label.bind("<B1-Motion>", do_drag)

        close_btn = tk.Button(
            header_frame, text="✕", bg="#007acc", fg="white",
            font=("Segoe UI", 10, "bold"), bd=0, activebackground="#005999",
            command=lambda: self._fire_command("reject_all")
        )
        close_btn.pack(side="right", padx=8)

        # ── Progress bar ──
        self._progress_label = tk.Label(
            root, text="Loading...",
            bg="#252526", fg="#808080",
            font=("Segoe UI", 9), anchor="w", padx=10, pady=4
        )
        self._progress_label.pack(fill="x")

        # ── Diff content area ──
        diff_frame = tk.Frame(root, bg="#1e1e1e", padx=8, pady=4)
        diff_frame.pack(fill="both", expand=True)

        self._diff_text = tk.Text(
            diff_frame, bg="#1e1e1e", fg="#d4d4d4",
            font=("Consolas", 10), wrap="word",
            borderwidth=0, highlightthickness=0,
            padx=8, pady=6, state="disabled",
            selectbackground="#264f78"
        )
        self._diff_text.pack(fill="both", expand=True)

        # Tag colours for diff lines
        self._diff_text.tag_configure("add", foreground="#b5cea8")
        self._diff_text.tag_configure("del", foreground="#f14c4c")
        self._diff_text.tag_configure("ctx", foreground="#808080")
        self._diff_text.tag_configure("header", foreground="#569cd6", font=("Consolas", 10, "bold"))

        # ── Status flash area ──
        self._status_label = tk.Label(
            root, text="",
            bg="#252526", fg="#4ec9b0",
            font=("Segoe UI", 9, "italic"), anchor="center", pady=2
        )
        self._status_label.pack(fill="x")

        # ── Button bar ──
        btn_frame = tk.Frame(root, bg="#252526", pady=6)
        btn_frame.pack(fill="x")

        buttons = [
            ("Accept [Y]", "#28a745", "accept"),
            ("Reject [N]", "#dc3545", "reject"),
            ("Accept All [A]", "#0e639c", "accept_all"),
            ("Reject All [R]", "#6c1d1d", "reject_all"),
            ("Explain [E]", "#6f42c1", "explain"),
        ]
        for label, colour, action in buttons:
            b = tk.Button(
                btn_frame, text=label, bg=colour, fg="white",
                font=("Segoe UI", 9, "bold"), bd=0, padx=8, pady=4,
                activebackground=colour,
                command=lambda a=action: self._fire_command(a)
            )
            b.pack(side="left", padx=3, expand=True)

        # ── Keyboard bindings ──
        root.bind("<y>", lambda e: self._fire_command("accept"))
        root.bind("<Y>", lambda e: self._fire_command("accept"))
        root.bind("<n>", lambda e: self._fire_command("reject"))
        root.bind("<N>", lambda e: self._fire_command("reject"))
        root.bind("<a>", lambda e: self._fire_command("accept_all"))
        root.bind("<A>", lambda e: self._fire_command("accept_all"))
        root.bind("<r>", lambda e: self._fire_command("reject_all"))
        root.bind("<R>", lambda e: self._fire_command("reject_all"))
        root.bind("<e>", lambda e: self._fire_command("explain"))
        root.bind("<E>", lambda e: self._fire_command("explain"))
        root.bind("<space>", lambda e: self._fire_command("skip"))
        root.bind("<Escape>", lambda e: self._fire_command("reject_all"))

        root.mainloop()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _update_display(self, hunk: DiffHunk, index: int, total: int) -> None:
        """Refresh the HUD with a new hunk."""
        if not self._root or not self._is_open:
            return

        # Update progress
        self._progress_label.configure(
            text=f"  📄 {hunk.file_path}  ·  Block {index} of {total}"
        )

        # Update diff content
        self._diff_text.configure(state="normal")
        self._diff_text.delete("1.0", "end")

        for line in hunk.diff_text.splitlines(keepends=True):
            if line.startswith("@@"):
                self._diff_text.insert("end", line, "header")
            elif line.startswith("+"):
                self._diff_text.insert("end", line, "add")
            elif line.startswith("-"):
                self._diff_text.insert("end", line, "del")
            else:
                self._diff_text.insert("end", line, "ctx")

        self._diff_text.configure(state="disabled")

        # Clear status
        self._status_label.configure(text="")

    def _flash_status(self, text: str) -> None:
        """Show a brief status message."""
        if self._status_label:
            self._status_label.configure(text=text)

    def _show_explanation(self, explanation: str) -> None:
        """Temporarily show an explanation in the diff area."""
        if not self._diff_text:
            return
        self._diff_text.configure(state="normal")
        self._diff_text.delete("1.0", "end")
        self._diff_text.insert("end", "💡 Explanation:\n\n", "header")
        self._diff_text.insert("end", explanation, "ctx")
        self._diff_text.configure(state="disabled")

    def _fire_command(self, action: str) -> None:
        """Route a HUD button press or hotkey to the coordinator callback."""
        if self.on_command:
            self.on_command(action, self._current_hunk_id)
