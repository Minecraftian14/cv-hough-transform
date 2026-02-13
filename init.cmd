@ECHO OFF
IF NOT EXIST ".venv" (
    ECHO No Virtual Environment Detected. Creating one...
    CALL virtualenv .venv
    CALL .venv\Scripts\activate.bat
    pip install -r requirements.txt
) ELSE (
    ECHO Loading Virtual Environment.
    CALL .venv\Scripts\activate.bat
    pip freeze> requirements.txt
)
