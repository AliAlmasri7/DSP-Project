import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import Task1
import Task4
import Task5
#import QuanTest1
#import QuanTest2


# ======================== Utility Function ========================
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


# ======================== Quantization Function ========================
def quantizeSignal(signal, flag, levels=0, numOfBits=0):
    """Quantize the given signal based on bits or levels, and compute quantization error."""
    if flag == 0:  # bits mode
        levels = int(np.power(2, numOfBits))

    minNum = np.min(signal[:, 1])
    maxNum = np.max(signal[:, 1])

    delta = (maxNum - minNum) / levels
    intervals = np.zeros((levels, 2))

    amountOfIncrease = minNum
    for i in range(levels):
        intervals[i, 0] = amountOfIncrease
        amountOfIncrease += delta
        intervals[i, 1] = amountOfIncrease

    midPoints = (intervals[:, 0] + intervals[:, 1]) / 2
    outputValues = np.zeros((signal.shape[0], 4), dtype=object)

    for i in range(signal.shape[0]):
        sample = signal[i, 1]
        for j in range(levels):
            if intervals[j, 0] <= sample < intervals[j, 1] or (j == levels - 1 and sample == intervals[j, 1]):
                intervalIndex = j
                break

        quantizedValue = midPoints[intervalIndex]
        error = quantizedValue - sample
        bits = int(np.ceil(np.log2(levels)))
        encodedValue = format(intervalIndex, f'0{bits}b')

        outputValues[i] = [intervalIndex + 1, encodedValue, quantizedValue, error]

    # Mean Squared Quantization Error
    squared_avg_error = np.mean(np.square(outputValues[:, 3].astype(float)))

    #QuanTest1.QuantizationTest1("Quan1_Out.txt",outputValues[:,1],outputValues[:,2])
    #QuanTest2.QuantizationTest2("Quan2_Out.txt",outputValues[:,0],outputValues[:,1],outputValues[:,2],outputValues[:,3])

    return outputValues, squared_avg_error


# ======================== GUI Application ========================
class SignalApp:
    def __init__(self, root):
       # self.current_signal = 
        self.current_signal = None
        self.loaded_file = None


        self.root = root
        self.root.title("DSP Signal Processing GUI")

        self.signals = []
        self.result = None

        # Frames
        control_frame = tk.Frame(root)
        control_frame.grid(row=0, column=0, sticky="ns", padx=10, pady=10)

        plot_frame = tk.Frame(root)
        plot_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        # Buttons in control frame
        tk.Label(control_frame, text="Operations", font=("Arial", 11, "bold")).grid(row=0, column=0, pady=(10, 5))
        tk.Button(control_frame, text="Load Signal", width=18, command=self.load_signal).grid(row=1, column=0, pady=3)
        tk.Button(control_frame, text="Add Signals", width=18, command=lambda: self.add_subtract(0)).grid(row=2, column=0, pady=3)
        tk.Button(control_frame, text="Subtract Signals", width=18, command=lambda: self.add_subtract(1)).grid(row=3, column=0, pady=3)
        tk.Button(control_frame, text="Scale Last Signal", width=18, command=self.scale_signal).grid(row=4, column=0, pady=3)
        tk.Button(control_frame, text="Shift Last Signal", width=18, command=self.shift_signal).grid(row=5, column=0, pady=3)
        tk.Button(control_frame, text="Fold Last Signal", width=18, command=self.fold_signal).grid(row=6, column=0, pady=3)
        tk.Button(control_frame, text="Display Last Signal", width=18, command=self.display_last_signal).grid(row=7, column=0, pady=3)

        # Signal generation
        tk.Label(control_frame, text="Signal Generation", font=("Arial", 11, "bold")).grid(row=8, column=0, pady=(10, 5))
        tk.Button(control_frame, text="Generate Sine Signal", width=18, command=lambda: self.generate_signal("sine")).grid(row=9, column=0, pady=3)
        tk.Button(control_frame, text="Generate Cosine Signal", width=18, command=lambda: self.generate_signal("cosine")).grid(row=10, column=0, pady=3)

        # Quantization section
        tk.Label(control_frame, text="Quantization", font=("Arial", 11, "bold")).grid(row=11, column=0, pady=(10, 5))
        tk.Button(control_frame, text="Quantize Signal", width=18, command=self.quantize_signal_gui).grid(row=12, column=0, pady=3)
        
        # Drivative section
        tk.Label(control_frame, text="Derivative", font=("Arial", 11, "bold")).grid(row=13, column=0, pady=(10, 5))
        tk.Button(control_frame, text="First Derivative", width=18, command=self.first_derivative_signal).grid(row=14, column=0, pady=3)
        tk.Button(control_frame, text="Second Derivative", width=18, command=self.second_derivative_signal).grid(row=15, column=0, pady=3)
        
        # Averaging section
        tk.Label(control_frame, text="Averaging", font=("Arial", 11, "bold")).grid(row=16, column=0, pady=(10, 5))
        tk.Button(control_frame, text="Moving Average", width=18, command=self.moving_average_signal).grid(row=17, column=0, pady=3)

        tk.Label(control_frame, text="Convolution", font=("Arial", 11, "bold")).grid(row=18, column=0, pady=(10, 5))
        tk.Button(control_frame, text="Convolve Signals", width=18, command=self.convolve_signals).grid(row=19, column=0, pady=3)

        # DFT & IDFT
        tk.Label(control_frame, text="DFT / IDFT", font=("Arial", 11, "bold")).grid(row=20, column=0, pady=(10, 5))
        tk.Button(control_frame, text="DFT", width=18, command=self.perform_dft).grid(row=21, column=0, pady=3)
        tk.Button(control_frame, text="IDFT", width=18, command=self.perform_idft).grid(row=22, column=0, pady=3)

        
        # Plot setup
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

    # ------------ Load Signal ------------
    def load_signal(self):
        file = filedialog.askopenfilename(title="Select Signal File", filetypes=[("Text Files", "*.txt")])
        if file:
            try:
                sig = Task1.readSignal(file)   # read first!
                self.current_signal = sig      # save last loaded signal
                self.loaded_file = file
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
        
        
     

   
    # ------------ Signal Generation ------------
    def generate_signal(self, sig_type):
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

                t_cont = np.linspace(0, T, 1000)
                t_disc = np.arange(0, T, 1/fs)
                theta = np.deg2rad(theta_deg)

                if sig_type == "sine":
                    y_cont = A * np.sin(2 * np.pi * f * t_cont + theta)
                    y_disc = A * np.sin(2 * np.pi * f * t_disc + theta)
                else:
                    y_cont = A * np.cos(2 * np.pi * f * t_cont + theta)
                    y_disc = A * np.cos(2 * np.pi * f * t_disc + theta)

                cont_signal = np.column_stack((t_cont, y_cont))
                disc_signal = np.column_stack((t_disc, y_disc))

                displaySignal(self.ax1, disc_signal, f"Discrete {sig_type.capitalize()} Signal")
                displaySignal(self.ax2, cont_signal, f"Continuous {sig_type.capitalize()} Signal", continuous=True)
                self.canvas.draw()
                self.result = disc_signal
                win.destroy()

            except ValueError:
                messagebox.showerror("Input Error", "Please fill all fields with valid numbers.")

        tk.Button(win, text="Generate", command=generate).grid(row=len(labels), columnspan=2, pady=10)
        
        
    
    def first_derivative_signal(self):
        if not self.loaded_file:
            messagebox.showwarning("Warning", "No file loaded yet!")
            return
        try:
            
            # Compute first derivative from file
            self.result = Task4.first_derivative(self.loaded_file)
            # Display derivative in second plot
            displaySignal(self.ax2, self.result, "First Derivative Signal")
            self.canvas.draw()
            messagebox.showinfo("Done", "First derivative computed and displayed in second plot. Check console for values.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute first derivative:\n{e}")
            
    def second_derivative_signal(self):
        if not self.loaded_file:
            messagebox.showwarning("Warning", "No signal loaded yet!")
            return
        try:
            self.result = Task4.second_derivative(self.loaded_file)
            # Keep original signal in first plot
            #displaySignal(self.ax1, self.result, "Original Signal")

            # Display derivative in second plot
            displaySignal(self.ax2, self.result, "Second Derivative (Sharpened Signal)")
            self.canvas.draw()

            messagebox.showinfo("Done", "Second derivative computed and displayed in second plot.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute Second derivative:\n{e}")      
            
            
    def moving_average_signal(self):
        if self.result is None:
            messagebox.showwarning("Warning", "No signal loaded yet!")
            return
        window = simpledialog.askinteger("Moving Average", "Enter window size:")
        if window is None:
            return
        try:
            ma_signal = Task4.moving_average(self.result, window)
            displaySignal(self.ax2, ma_signal, f"Moving Average (window={window})")
            self.canvas.draw()
            messagebox.showinfo("Done", f"Moving average computed with window size {window}. Check console for values.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def convolve_signals(self):
        try:
            file1 = filedialog.askopenfilename(title="Select first signal file", filetypes=[("Text Files", "*.txt")])
            if not file1:
                return
            file2 = filedialog.askopenfilename(title="Select second signal file", filetypes=[("Text Files", "*.txt")])
            if not file2:
                return

            # Read both signals using your readSignal function
            sig1 = Task1.readSignal(file1)
            sig2 = Task1.readSignal(file2)

            # Perform convolution (note: now we pass arrays, not file names)
            convolved_signal = Task4.convolution(sig1, sig2)

            # Display in second plot
            displaySignal(self.ax2, convolved_signal, "Convolved Signal")
            self.canvas.draw()

            messagebox.showinfo("Success", "Convolution completed successfully! Check console for details.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to convolve signals:\n{e}")
            
            
    def perform_dft(self):
        try:
            if self.result is None:
                messagebox.showerror("Error", "Load or generate a signal first!")
                return

            # Ask for sampling frequency
            fs = simpledialog.askfloat("Sampling Frequency",
                                    "Enter sampling frequency (Hz):",
                                    minvalue=0.1)
            if fs is None:
                return

            # Perform DFT using last signal
            freq, X = Task5.DFT_or_IDFT(self.result, inverse=False, fs=fs)

            amplitude = np.abs(X)
            phase = np.angle(X)

            # Plot amplitude spectrum on ax1
            displaySignal(self.ax1, np.column_stack((freq, amplitude)),
                        "Amplitude Spectrum (|X[k]|)")
            
            # Plot phase spectrum on ax2
            displaySignal(self.ax2, np.column_stack((freq, phase)),
                        "Phase Spectrum (angle(X[k]))")

            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute DFT\n{e}")
   



    def perform_idft(self):
        try:
            if self.result is None:
                messagebox.showerror("Error", "Perform DFT first or load a magnitude/phase file!")
                return

            # Ask the user if the input is in polar form
            polar = messagebox.askyesno("Input Type", "Is the input in Magnitude/Phase (polar) form?")

            # If the input is polar, we can skip asking for fs (optional)
            if polar:
                # Convert last loaded signal if needed
                signal_to_use = self.result
                freq, x_rec = Task5.DFT_or_IDFT(signal_to_use, inverse=True, polar=True)
            else:
                # Ask for sampling frequency
                fs = simpledialog.askfloat("Sampling Frequency",
                                        "Enter sampling frequency (Hz):",
                                        minvalue=0.1)
                if fs is None:
                    return
                freq, x_rec = Task5.DFT_or_IDFT(self.result, inverse=True, fs=fs)

            # Take real part for plotting (IDFT result)
            x_real = np.real(x_rec)
            n = np.arange(len(x_real))
            reconstructed = np.column_stack((n, x_real))

            # Plot on ax2
            displaySignal(self.ax2, reconstructed, "Reconstructed Signal (IDFT)")
            self.canvas.draw()

            messagebox.showinfo("Done", "IDFT computed and displayed in second plot.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to perform IDFT\n{e}")




        

    
    # ------------ Quantization GUI ------------
    def quantize_signal_gui(self):
        if self.result is None:
            messagebox.showwarning("Warning", "No signal to quantize!")
            return

        win = tk.Toplevel(self.root)
        win.title("Quantize Signal")

        tk.Label(win, text="Choose Quantization Mode:").grid(row=0, column=0, padx=5, pady=5)
        mode_var = tk.StringVar(value="bits")
        tk.Radiobutton(win, text="Number of Bits", variable=mode_var, value="bits").grid(row=0, column=1)
        tk.Radiobutton(win, text="Number of Levels", variable=mode_var, value="levels").grid(row=0, column=2)

        value_entry = tk.Entry(win)
        value_entry.grid(row=1, column=1, padx=5, pady=5)
        tk.Label(win, text="Enter Value:").grid(row=1, column=0, padx=5, pady=5)

        show_index = tk.BooleanVar()
        show_error = tk.BooleanVar()
        tk.Checkbutton(win, text="Show Interval Index", variable=show_index).grid(row=2, column=0, padx=5)
        tk.Checkbutton(win, text="Show Quantization Error", variable=show_error).grid(row=2, column=1, padx=5)

        result_box = ttk.Treeview(win, columns=("Index", "Encoded", "Quantized", "Error"), show="headings", height=10)
        for col in ("Index", "Encoded", "Quantized", "Error"):
            result_box.heading(col, text=col)
            result_box.column(col, width=100)
        result_box.grid(row=4, columnspan=3, pady=10)

        mse_label = tk.Label(win, text="Mean Squared Error: -")
        mse_label.grid(row=5, columnspan=3, pady=5)

        def run_quantization():
            try:
                val = int(value_entry.get())
                flag = 0 if mode_var.get() == "bits" else 1
                output, mse = quantizeSignal(self.result, flag, numOfBits=val if flag == 0 else 0, levels=val if flag == 1 else 0)

                result_box.delete(*result_box.get_children())
                for row in output:
                    idx, enc, quant, err = row
                    show = []
                    if show_index.get():
                        show.append(str(idx))
                    else:
                        show.append("")
                    show.append(enc)
                    show.append(f"{quant:.3f}")
                    if show_error.get():
                        show.append(f"{err:.4f}")
                    else:
                        show.append("")
                    result_box.insert("", "end", values=show)

                mse_label.config(text=f"Mean Squared Error: {mse:.6f}")

                # Plot quantized digital signal
                quantized_signal = np.column_stack((self.result[:, 0], output[:, 2].astype(float)))
                displaySignal(self.ax1, quantized_signal, "Quantized Digital Signal")
                self.canvas.draw()

            except ValueError:
                messagebox.showerror("Error", "Enter a valid integer for bits/levels.")

        tk.Button(win, text="Quantize", command=run_quantization).grid(row=3, columnspan=3, pady=10)
        
        
                
        
                


# ---------- Run Application ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = SignalApp(root)
    root.mainloop()
