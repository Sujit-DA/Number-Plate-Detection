from datetime import datetime, timedelta
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import sqlite3
import math
import re
import os
import json
import numpy as np
import logging


# Initialize logging
logging.basicConfig(filename='parking_system.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the YOLO model and PaddleOCR
model = YOLO("best.pt")
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

# Cooldown period to prevent duplicate detection
COOLDOWN_PERIOD = timedelta(seconds=1000000)

# Database initialization
def initialize_database():
    """Initialize the SQLite database and create the table if it does not exist."""
    conn = sqlite3.connect('licensePlatesDatabase.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS LicensePlates(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            license_plate TEXT NOT NULL UNIQUE,
            vehicle_type TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            bill REAL,
            status TEXT DEFAULT 'In',
            CHECK(exit_time IS NULL OR exit_time >= entry_time)
        )
    ''')
    conn.commit()
    conn.close()

initialize_database()

def preprocess_image(cropped_frame):
    """Preprocess image for OCR."""
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def is_valid_plate(plate):
    """Validate Nepali embossed number plates."""
    pattern = re.compile(r'^[AB][A-Z]{2}\d{4}$')
    return bool(pattern.match(plate))

def paddle_ocr(frame, x1, y1, x2, y2, confidence_threshold=0.6):
    """Perform OCR on a license plate."""
    cropped_frame = frame[y1:y2, x1:x2]
    preprocessed_frame = preprocess_image(cropped_frame)
    result = ocr.ocr(preprocessed_frame, det=False, rec=True, cls=False)
    text = ""
    for r in result:
        scores = r[0][1]
        if not np.isnan(scores) and scores > confidence_threshold:
            text = r[0][0]
    # Clean up the detected text
    pattern = re.compile(r'[\W]')
    text = pattern.sub('', text).replace("O", "0")
    if is_valid_plate(text):
        return text
    return None


def save_to_database(plate, vehicle_type, entry_time, exit_time=None, bill=None):
    try:
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()

        if exit_time:
            # Check if the vehicle is already in the database with 'In' status
            cursor.execute('''
                SELECT entry_time FROM LicensePlates 
                WHERE license_plate = ? AND status = 'In'
            ''', (plate,))
            record = cursor.fetchone()

            if not record:
                logging.warning(f"Exit attempt for non-existent or already exited vehicle: {plate}")
                return "Vehicle not found or already exited."

            # Update the record with exit time and bill
            cursor.execute('''
                UPDATE LicensePlates
                SET exit_time = ?, bill = ?, status = 'Out'
                WHERE license_plate = ? AND status = 'In'
            ''', (exit_time, bill, plate))
            conn.commit()

            # Generate invoice
            invoice_message = generate_invoice(plate)
            logging.info(f"Invoice generated for {plate}: {invoice_message}")

            return "Exit recorded and invoice generated successfully."
        else:
            # Insert a new entry for the vehicle
            cursor.execute('''
                INSERT INTO LicensePlates(license_plate, vehicle_type, entry_time, exit_time, bill, status)
                VALUES (?, ?, ?, ?, ?, 'In')
            ''', (plate, vehicle_type, entry_time, exit_time, bill))
            logging.info(f"New vehicle entry: {plate}")
            conn.commit()
            return "Entry recorded successfully."

    except sqlite3.IntegrityError:
        logging.warning(f"Duplicate license plate entry attempted: {plate}")
        return "Duplicate entry detected."
    except Exception as e:
        logging.error(f"Error saving to database: {e}")
        return "An error occurred while saving to the database."
    finally:
        conn.close()



def save_to_database(plate, vehicle_type, entry_time, exit_time=None, bill=None):
    """Save data to the SQLite database."""
    try:
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()

        # Check if the plate is already in the database with status 'In'
        cursor.execute('''
            SELECT entry_time FROM LicensePlates WHERE license_plate = ? AND status = 'In'
        ''', (plate,))
        record = cursor.fetchone()

        if record:
            # Exit event: Update exit time and calculate bill
            entry_time = datetime.fromisoformat(record[0])
            current_time = datetime.now()
            bill = calculate_bill(entry_time, current_time, vehicle_type)

            cursor.execute('''
                UPDATE LicensePlates
                SET exit_time = ?, bill = ?, status = 'Out'
                WHERE license_plate = ? AND status = 'In'
            ''', (current_time.isoformat(), bill, plate))

        else:
            # Entry event: Add to database
            cursor.execute('''
                INSERT INTO LicensePlates(license_plate, vehicle_type, entry_time, exit_time, bill, status)
                VALUES (?, ?, ?, ?, ?, 'In')
            ''', (plate, vehicle_type, entry_time, None, None))
            logging.info(f"New vehicle entry: {plate}")

        conn.commit()
    except sqlite3.IntegrityError:
        logging.warning(f"Duplicate license plate entry attempted: {plate}")
    finally:
        conn.close()

def calculate_bill(entry_time, exit_time, vehicle_type):
    """Calculate the parking bill."""
    # duration = (exit_time - entry_time).total_seconds() / 1800  # Half-hour intervals
    # rate = 30 if vehicle_type == "Two-Wheeler" else 50
    # return math.ceil(duration) * rate
    if vehicle_type == "Two-Wheeler":
        return 20.0
    elif vehicle_type == "Four-Wheeler":
        return 35.0
    else:
        logging.warning(f"Unknown vehicle type: {vehicle_type}")
        return 0.0

# Add this function to generate and print the bill
def generate_bill(plate):
    """
    Generate a parking bill for a specific license plate.

    Args:
        plate (str): The license plate number.

    Returns:
        str: A formatted bill or an error message.
    """
    try:
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()

        # Fetch the record for the given license plate
        cursor.execute('''
            SELECT vehicle_type, entry_time, exit_time, bill
            FROM LicensePlates
            WHERE license_plate = ? AND status = 'Out'
        ''', (plate,))
        record = cursor.fetchone()

        if not record:
            logging.warning(f"No completed parking record found for plate: {plate}")
            return f"No billing details found for license plate: {plate}"

        vehicle_type, entry_time_str, exit_time_str, bill = record

        # Format the bill
        bill_details = f"""
        ----------------------------------------
                     PARKING BILL
        ----------------------------------------
        License Plate: {plate}
        Vehicle Type:  {vehicle_type}
        Entry Time:    {entry_time_str}
        Exit Time:     {exit_time_str}
        Total Bill:    Rs. {bill:.2f}
        ----------------------------------------
            Thank you for using our service!
        ----------------------------------------
        """
        print(bill_details)
        return bill_details

    except Exception as e:
        logging.error(f"Error generating bill: {e}")
        return "An error occurred while generating the bill."
    finally:
        conn.close()


def generate_invoice(plate):
    """
    Generate a parking invoice for a specific license plate and save it as a JSON file.

    Args:
        plate (str): The license plate number.

    Returns:
        str: Path to the generated JSON invoice or an error message.
    """
    try:
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()

        # Fetch the record for the given license plate
        cursor.execute('''
            SELECT vehicle_type, entry_time, exit_time, bill
            FROM LicensePlates
            WHERE license_plate = ? AND status = 'Out'
        ''', (plate,))
        record = cursor.fetchone()

        if not record:
            logging.warning(f"No completed parking record found for plate: {plate}")
            return f"No billing details found for license plate: {plate}"

        vehicle_type, entry_time_str, exit_time_str, bill = record

        # Create invoice data
        invoice_data = {
            "license_plate": plate,
            "vehicle_type": vehicle_type,
            "entry_time": entry_time_str,
            "exit_time": exit_time_str,
            "total_bill": bill,
            "invoice_date": datetime.now().isoformat(),
            "message": "Thank you for using our parking service!"
        }

        # Save invoice as JSON file
        invoice_filename = f"invoice_{plate}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        with open(invoice_filename, 'w') as invoice_file:
            json.dump(invoice_data, invoice_file, indent=4)

        logging.info(f"Invoice generated: {invoice_filename}")
        return f"Invoice saved to {invoice_filename}"

    except Exception as e:
        logging.error(f"Error generating invoice: {e}")
        return "An error occurred while generating the invoice."
    finally:
        conn.close()


# Video source: Set to True for live webcam or provide a video file path
use_live_camera = False  # Set to True to use live camera
video_path = "video30.mp4"  

# Select video source
if use_live_camera:
    logging.info("Using live webcam as the video source.")
    cap = cv2.VideoCapture(0)  # Open live webcam (0 is the default camera index)
else:
    if not os.path.exists(video_path):
        logging.error(f"Video file {video_path} does not exist.")
        raise FileNotFoundError(f"Video file {video_path} not found.")
    logging.info(f"Using video file: {video_path} as the video source.")
    cap = cv2.VideoCapture(video_path)  # Load video file

# Check if the video source is open correctly
if not cap.isOpened():
    logging.error(f"Failed to open video source {video_path if not use_live_camera else 'webcam'}.")
    raise Exception(f"Unable to open video source {video_path if not use_live_camera else 'webcam'}.")

# Active vehicles dictionary
active_vehicles = {}

# Main loop for processing video frames
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.45)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = paddle_ocr(frame, x1, y1, x2, y2)

                if label:
                    current_time = datetime.now()
                    vehicle_type = "Two-Wheeler" if label[0] == 'A' else "Four-Wheeler" if label[0] == 'B' else "Unknown"

                    # Check if the vehicle is exiting
                    conn = sqlite3.connect('licensePlatesDatabase.db')
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT status FROM LicensePlates WHERE license_plate = ?
                    ''', (label,))
                    record = cursor.fetchone()
                    conn.close()

                    if record and record[0] == 'In':
                        # Vehicle exiting, save exit time and calculate bill
                        save_to_database(label, vehicle_type, None, current_time.isoformat())
                        generate_bill(label)  # Generate and print the bill
                    else:
                        # New entry
                        save_to_database(label, vehicle_type, current_time.isoformat())

                # Draw bounding box and label in blue
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue rectangle
                cv2.putText(frame, label if label else "Invalid", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Blue text

        # Display the video frame
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    logging.error(f"Unexpected error during video processing: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()