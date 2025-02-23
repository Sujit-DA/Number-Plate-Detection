import sqlite3
import logging
import math
from datetime import datetime, timedelta

# Cooldown period to prevent duplicate detection
COOLDOWN_PERIOD = timedelta(seconds=2)

def save_to_database(plate, vehicle_type, entry_time, exit_time=None, bill=None):
    """
    Save data to the SQLite database, updating exit_time and bill if the vehicle exits.
    
    Args:
        plate (str): License plate number.
        vehicle_type (str): Type of vehicle (e.g., "Two-Wheeler", "Four-Wheeler").
        entry_time (str): ISO format timestamp for entry time.
        exit_time (str, optional): ISO format timestamp for exit time. Defaults to None.
        bill (float, optional): Calculated parking bill. Defaults to None.
        
    Returns:
        str: Status message indicating success or failure.
    """
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

            # Calculate the bill
            entry_time_dt = datetime.fromisoformat(record[0])
            exit_time_dt = datetime.fromisoformat(exit_time)
            bill = calculate_bill(entry_time_dt, exit_time_dt, vehicle_type)

            # Update the record with exit time and bill
            cursor.execute('''
                UPDATE LicensePlates
                SET exit_time = ?, bill = ?, status = 'Out'
                WHERE license_plate = ? AND status = 'In'
            ''', (exit_time, bill, plate))
            logging.info(f"Vehicle exit recorded: {plate}, Bill: {bill}")
            conn.commit()
            return "Exit recorded successfully."
        else:
            # Check if the vehicle is already in the database with 'In' status
            cursor.execute('''
                SELECT entry_time, exit_time FROM LicensePlates 
                WHERE license_plate = ? AND status = 'Out'
            ''', (plate,))
            record = cursor.fetchone()

            if record:
                # Check if cooldown period has passed
                exit_time_dt = datetime.fromisoformat(record[1])
                if datetime.fromisoformat(entry_time) - exit_time_dt < COOLDOWN_PERIOD:
                    logging.warning(f"Cooldown period not passed for vehicle: {plate}")
                    return "Cooldown period not passed."

            # Insert a new entry for the vehicle
            cursor.execute('''
                INSERT INTO LicensePlates(license_plate, vehicle_type, entry_time, exit_time, bill, status)
                VALUES (?, ?, ?, ?, ?, 'In')
            ''', (plate, vehicle_type, entry_time, None, None))
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

def calculate_bill(entry_time, exit_time, vehicle_type):
    """Calculate the parking bill."""
    duration = (exit_time - entry_time).total_seconds() / 3600  # Hours
    rate = 20 if vehicle_type == "Two-Wheeler" else 35
    return math.ceil(duration) * rate