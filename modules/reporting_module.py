import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import json

class ReportingModule:
    def __init__(self):
        self.session_log = []
        self.total_pieces_processed = 0
        self.grade_counts = {0: 0, 1: 0, 2: 0, 3: 0} # Grade 0 for perfect wood
        self.last_report_path = None

    def log_action(self, message):
        """Log actions to file with timestamp."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp} - {message}\n"
            
            # Ensure the logs directory exists
            log_dir = "wood_sorting_app/logs"
            os.makedirs(log_dir, exist_ok=True)
            
            with open(os.path.join(log_dir, "activity_log.txt"), "a") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Error logging action: {e}")

    def finalize_grading_log(self, final_grade, all_measurements, piece_number):
        """Central function to log piece details for reporting."""
        defects_for_log = []
        if all_measurements:
            defect_summary = {}
            for defect_type, size_mm, percentage in all_measurements:
                if defect_type not in defect_summary:
                    defect_summary[defect_type] = {'count': 0, 'sizes_mm': []}
                defect_summary[defect_type]['count'] += 1
                defect_summary[defect_type]['sizes_mm'].append(f"{size_mm:.1f}")

            for defect_type, data in defect_summary.items():
                defects_for_log.append({
                    'type': defect_type.replace('_', ' '),
                    'count': data['count'],
                    'sizes': ', '.join(data['sizes_mm'])
                })
        
        log_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "piece_number": piece_number,
            "final_grade": final_grade,
            "defects": defects_for_log
        }
        self.session_log.append(log_entry)

    def generate_report(self):
        """Generate a comprehensive report (TXT and PDF)."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_filename = f"report_{timestamp}"
        
        # Ensure the reports directory exists
        report_dir = "wood_sorting_app/reports"
        os.makedirs(report_dir, exist_ok=True)

        txt_filepath = os.path.join(report_dir, f"{base_filename}.txt")
        pdf_filepath = os.path.join(report_dir, f"{base_filename}.pdf")
        
        # --- Build Report Content ---
        content = f"--- SS-EN 1611-1 Wood Sorting Report ---\n"
        content += f"Generated at: {timestamp}\n\n"
        content += "--- Session Summary ---\n"
        content += f"Total Pieces Processed: {self.total_pieces_processed}\n"
        content += f"Grade Perfect (No Defects): {self.grade_counts.get(0, 0)}\n"
        content += f"Grade G2-0/G2-1 (Good Quality): {self.grade_counts.get(1, 0)}\n"  
        content += f"Grade G2-2/G2-3 (Fair Quality): {self.grade_counts.get(2, 0)}\n"
        content += f"Grade G2-4 (Poor Quality): {self.grade_counts.get(3, 0)}\n"
        
        content += "\n\n--- Individual Piece Log ---\n"
        if not self.session_log:
            content += "No pieces were processed in this session.\n"
        else:
            for entry in self.session_log:
                content += f"\nPiece #{entry['piece_number']}: Grade {entry['final_grade']}\n"
                if not entry['defects']:
                    content += "  - No defects detected.\n"
                else:
                    for defect in entry['defects']:
                        content += f"  - Defect: {defect['type']}, Count: {defect['count']}, Sizes (mm): {defect['sizes']}\n"

        # Save TXT report
        try:
            with open(txt_filepath, 'w') as f:
                f.write(content)
            print(f"SS-EN 1611-1 report generated: {txt_filepath}")
        except Exception as e:
            print(f"Error generating TXT report: {e}")
            # In a real app, you might send this error to the GUI

        # Generate PDF Report
        try:
            c = canvas.Canvas(pdf_filepath, pagesize=letter)
            width, height = letter
            
            c.setFont("Helvetica-Bold", 16)
            c.drawCentredString(width / 2.0, height - 1*inch, "SS-EN 1611-1 Wood Sorting System Report")
            c.setFont("Helvetica", 12)
            
            text = c.beginText(1*inch, height - 1.5*inch)
            text.textLine(f"Generated at: {timestamp}")
            text.textLine("")
            text.setFont("Helvetica-Bold", 12)
            text.textLine("Session Summary")
            text.setFont("Helvetica", 12)
            text.textLine(f"Total Pieces Processed: {self.total_pieces_processed}")
            text.textLine(f"Grade Perfect (No Defects): {self.grade_counts.get(0, 0)}")
            text.textLine(f"Grade G2-0/G2-1 (Good Quality): {self.grade_counts.get(1, 0)}")
            text.textLine(f"Grade G2-2/G2-3 (Fair Quality): {self.grade_counts.get(2, 0)}")
            text.textLine(f"Grade G2-4 (Poor Quality): {self.grade_counts.get(3, 0)}")
            text.textLine("")
            text.textLine("")
            text.setFont("Helvetica-Bold", 12)
            text.textLine("Individual Piece Log")
            text.setFont("Helvetica", 12)

            if not self.session_log:
                text.textLine("No pieces were processed in this session.")
            else:
                for entry in self.session_log:
                    if text.getY() < 2 * inch:
                        c.drawText(text)
                        c.showPage()
                        c.setFont("Helvetica", 12)
                        text = c.beginText(1*inch, height - 1*inch)

                    text.textLine("")
                    text.setFont("Helvetica-Bold", 10)
                    text.textLine(f"Piece #{entry['piece_number']}: Grade {entry['final_grade']}")
                    text.setFont("Helvetica", 10)
                    if not entry['defects']:
                        text.textLine("  - No defects detected.")
                    else:
                        for defect in entry['defects']:
                            text.textLine(f"  - Defect: {defect['type']}, Count: {defect['count']}, Sizes (mm): {defect['sizes']}")
            
            c.drawText(text)
            c.save()
            print(f"SS-EN 1611-1 PDF report generated: {pdf_filepath}")
            
            self.last_report_path = pdf_filepath

        except Exception as e:
            print(f"Error generating PDF report: {e}")
            # In a real app, you might send this error to the GUI
            
        # Reset the session log after generating the report
        self.session_log = []
        print("Session log has been cleared for the next report.")

    def save_detection_frame(self, camera_name, frame):
        """Save a detection frame as image file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{camera_name}_{timestamp}.jpg"
            
            # Ensure the detection_frames directory exists
            frame_dir = "wood_sorting_app/detection_frames"
            os.makedirs(frame_dir, exist_ok=True)

            filepath = os.path.join(frame_dir, filename)
            
            # Convert from RGB back to BGR for OpenCV if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            cv2.imwrite(filepath, frame_bgr)
            print(f"📸 Saved detection frame: {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving detection frame: {e}")