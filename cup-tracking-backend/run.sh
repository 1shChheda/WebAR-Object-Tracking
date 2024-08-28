#!/bin/bash
# After entering the "cup-tracking-backend" directory, enter the command:
    # cup_tracking_env\Scripts\activate
    # & then, the following command:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
