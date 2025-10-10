import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import Task1  # your DSP functions (keep this import)


def displaySignal(ax, samples, title="Signal", continuous=False):
    """Display signal as discrete (stem) or continuous (plot)."""
    ax.clear()
    if continuous:
        ax.plot(samples[:, 0], samples[:, 1], label="Continuous")
    else:
        ax.stem(samples[:, 0], samples[:, 1], linefmt='b-', markerfmt='bo', basefmt='r-')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True)


class SignalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DSP Signal Processing GUI")

        self.signals = []
        self.result = None

        # Create frames
        control_frame = tk.Frame(root)
        control_frame.grid(row=0, column=0, sticky="ns", padx=10, pady=10)

        plot_frame = tk.Frame(root)
        plot_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # Expand the right plot area when resizing
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        # Buttons in control frame
        tk.Button(control_frame, text="Load Signal", width=18, command=self.load_signal).grid(row=0, column=0, pady=3)
        tk.Button(control_frame, text="Add Signals", width=18, command=lambda: self.add_subtract(0)).grid(row=1, column=0, pady=3)
        tk.Button(control_frame, text="Subtract Signals", width=18, command=lambda: self.add_subtract(1)).grid(row=2, column=0, pady=3)
        tk.Button(control_frame, text="Scale Last Signal", width=18, command=self.scale_signal).grid(row=3, column=0, pady=3)
        tk.Button(control_frame, text="Shift Last Signal", width=18, command=self.shift_signal).grid(row=4, column=0, pady=3)
        tk.Button(control_frame, text="Fold Last Signal", width=18, command=self.fold_signal).grid(row=5, column=0, pady=3)
        tk.Button(control_frame, text="Display Last Signal", width=18, command=self.display_last_signal).grid(row=6, column=0, pady=3)

        # Separator label
        tk.Label(control_frame, text="Signal Generation", font=("Arial", 11, "bold")).grid(row=7, column=0, pady=(10, 5))
        tk.Button(control_frame, text="Generate Sine Signal", width=18, command=lambda: self.generate_signal("sine")).grid(row=8, column=0, pady=3)
        tk.Button(control_frame, text="Generate Cosine Signal", width=18, command=lambda: self.generate_signal("cosine")).grid(row=9, column=0, pady=3)

        # Matplotlib plots in plot_frame
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)


    # ------------ Signal Loading ------------
    def load_signal(self):
        file = filedialog.askopenfilename(title="Select Signal File", filetypes=[("Text Files", "*.txt")])
        if file:
            try:
                sig = Task1.readSignal(file)
                self.signals.append(sig)
                self.result = sig
                messagebox.showinfo("Loaded", f"Signal loaded with {sig.shape[0]} samples.")
                displaySignal(self.ax1, sig, f"Loaded Signal {len(self.signals)}")
                self.canvas.draw()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    # ------------ Signal Operations ------------
    def add_subtract(self, op):
        if len(self.signals) < 2:
            messagebox.showwarning("Warning", "Load at least 2 signals first!")
            return
        self.result = Task1.addSubtractSignals(*self.signals, operation=op)
        title = "Added Signals" if op == 0 else "Subtracted Signals"
        displaySignal(self.ax1, self.result, title)
        self.canvas.draw()

    def scale_signal(self):
        if self.result is None:
            messagebox.showwarning("Warning", "No signal to scale!")
            return
        factor = simpledialog.askfloat("Scale", "Enter scale factor:")
        if factor is not None:
            self.result = Task1.multiplySignal(self.result.copy(), factor)
            displaySignal(self.ax1, self.result, f"Scaled Signal (x{factor})")
            self.canvas.draw()

    def shift_signal(self):
        if self.result is None:
            messagebox.showwarning("Warning", "No signal to shift!")
            return
        k = simpledialog.askinteger("Shift", "Enter shift value (integer):")
        direction = simpledialog.askstring("Shift", "Enter '+' for left, '-' for right:")
        if k is not None and direction in ['+', '-']:
            self.result = Task1.shiftSignal(self.result, k, direction)
            displaySignal(self.ax1, self.result, f"Shifted Signal ({direction}{k})")
            self.canvas.draw()

    def fold_signal(self):
        if self.result is None:
            messagebox.showwarning("Warning", "No signal to fold!")
            return
        self.result = Task1.foldSignal(self.result)
        displaySignal(self.ax1, self.result, "Folded Signal (time reversal)")
        self.canvas.draw()

    def display_last_signal(self):
        if self.result is None:
            messagebox.showwarning("Warning", "No signal loaded yet!")
            return
        displaySignal(self.ax1, self.result, "Last Signal")
        self.canvas.draw()

    # ------------ New: Generate Sine/Cosine ------------
    def generate_signal(self, sig_type):
        # Open a small parameter input window
        win = tk.Toplevel(self.root)
        win.title(f"Generate {sig_type.capitalize()} Signal")

        entries = {}
        labels = ["Amplitude (A)", "Phase Shift (θ degrees)", "Analog Frequency (Hz)", "Sampling Frequency (Hz)", "Duration (seconds)"]
        for i, text in enumerate(labels):
            tk.Label(win, text=text).grid(row=i, column=0, padx=5, pady=3)
            entry = tk.Entry(win)
            entry.grid(row=i, column=1, padx=5, pady=3)
            entries[text] = entry

        def generate():
            try:
                A = float(entries["Amplitude (A)"].get())
                theta_deg = float(entries["Phase Shift (θ degrees)"].get())
                f = float(entries["Analog Frequency (Hz)"].get())
                fs = float(entries["Sampling Frequency (Hz)"].get())
                T = float(entries["Duration (seconds)"].get())

                if fs < 2 * f:
                    messagebox.showwarning("Sampling Warning", "Sampling frequency must be at least 2× analog frequency (Nyquist theorem).")
                    return

                t_cont = np.linspace(0, T, 1000)  # for continuous
                t_disc = np.arange(0, T, 1/fs)    # for discrete

                theta = np.deg2rad(theta_deg)

                if sig_type == "sine":
                    y_cont = A * np.sin(2 * np.pi * f * t_cont + theta)
                    y_disc = A * np.sin(2 * np.pi * f * t_disc + theta)
                else:
                    y_cont = A * np.cos(2 * np.pi * f * t_cont + theta)
                    y_disc = A * np.cos(2 * np.pi * f * t_disc + theta)

                cont_signal = np.column_stack((t_cont, y_cont))
                disc_signal = np.column_stack((t_disc, y_disc))

                # Plot both
                displaySignal(self.ax1, disc_signal, f"Discrete {sig_type.capitalize()} Signal")
                displaySignal(self.ax2, cont_signal, f"Continuous {sig_type.capitalize()} Signal", continuous=True)
                self.canvas.draw()

                win.destroy()

            except ValueError:
                messagebox.showerror("Input Error", "Please fill all fields with valid numbers.")

        tk.Button(win, text="Generate", command=generate).grid(row=len(labels), columnspan=2, pady=10)


# ---------- Run Application ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = SignalApp(root)
    root.mainloop()
