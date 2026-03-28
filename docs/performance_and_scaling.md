# Performance and Scaling Guide

This document outlines the system's strategy for performance, memory management, and scaling to multiple trading pairs, even on hardware with limited resources like a personal computer with 6GB of VRAM.

## The Core Principle: Sequential Processing

The key to running a sophisticated AI model across many symbols (e.g., 10 or more pairs) on consumer hardware is to **avoid loading everything into memory at once**.

This project implements a **sequential, single-symbol processing loop** within the `run_signals.py` script.

## How It Works

The main scanning job (`scan_job`) iterates through the list of symbols defined in your `config/settings.yaml`. For each symbol in the list, it performs the following steps:

1.  **Load Data**: Fetches the historical data for **only one symbol** (e.g., "BTCUSDT").
2.  **Load Model & Infer**: The AI model performs its analysis (inference) on that single symbol's data.
    -   **`torch.no_grad()`**: This PyTorch context manager is used during inference. It disables gradient calculations, which significantly reduces memory consumption and speeds up the process.
3.  **Generate Signal**: A signal is generated (or not) for that symbol.
4.  **Clean Up Memory**: This is the most critical step for scaling.
    -   **`torch.cuda.empty_cache()`**: If a GPU is being used (`device: "cuda"`), this command is called. It releases all unoccupied cached memory back to the GPU, ensuring that the memory footprint from analyzing the previous symbol is wiped clean before starting the next one.
    -   All data-related variables for the symbol (like the large data DataFrame) go out of scope at the end of the loop iteration and are garbage collected by Python.
5.  **Repeat**: The loop then moves to the next symbol (e.g., "ETHUSDT") and repeats the process with a clean memory state.

## Benefits of This Approach

-   **Low VRAM Usage**: The VRAM required is only enough to hold the data and model for a single symbol at a time, not all ten. This is why the system can run smoothly on a GPU with as little as 6GB of VRAM.
-   **Scalability**: This design can be scaled to 20, 50, or even 100 symbols without increasing the peak VRAM usage. The only trade-off is that each scan cycle will take longer to complete.
-   **Stability**: It prevents "Out of Memory" errors that would crash the application if it tried to load all data simultaneously.

This memory-conscious design ensures that the AI can run 24/7 in a stable and efficient manner on accessible, everyday hardware.
