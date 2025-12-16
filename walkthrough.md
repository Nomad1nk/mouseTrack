# Walkthrough - Fix Malfunction in eye_mouse.py

## Issue
The user reported that `eye_mouse.py` was "malfunctioning".
Investigation revealed that the script used `pyautogui.sleep()` which blocks the main thread, causing the video feed to freeze whenever a click or button interaction occurred. This made the application feel unresponsive or broken.

## Changes
I refactored `eye_mouse.py` to use non-blocking logic:
-   **Removed `pyautogui.sleep()`**: Replaced with `time.time()` checks.
-   **Added `click_cooldown`**: Prevents multiple clicks from registering instantly (set to 1.0 second).
-   **Improved Button Logic**: Button interactions now respect the cooldown and do not freeze the frame.
-   **Set `pyautogui.PAUSE = 0`**: Ensures `pyautogui` actions are faster.

## Verification
I ran the script `python eye_mouse.py` and confirmed that it starts without crashing. The removal of blocking sleeps should ensure smooth video playback even during interactions.

## Fix 'SymbolDatabase' Error
### Issue
The user reported `Error: 'SymbolDatabase' object has no attribute 'GetPrototype'` when running `virtual_mouse.py`.
This is caused by an incompatibility between `mediapipe` and newer versions of `protobuf` (4.x+).

### Changes
-   **Downgraded `protobuf`**: Installed `protobuf==3.20.3`.
-   **Updated `requirements.txt`**: Pinned `protobuf==3.20.3` to prevent future issues.

### Verification
-   Ran `python virtual_mouse.py` and confirmed it starts successfully without the `AttributeError`.
