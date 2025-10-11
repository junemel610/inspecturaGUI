/*
  Master Controller for Automated Wood Sorting System
  Stable Scan Phase with Proper Reset and Calibrated Speed
*/

#include <Servo.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// --- Pin Definitions ---
const int IR_SENSOR_PIN = 11;
const int SERVO_1_PIN = 2;
const int SERVO_2_PIN = 3;
const int SERVO_3_PIN = 4;
const int SERVO_4_PIN = 5;
const int STEPPER_ENA_PIN = 8;
const int STEPPER_DIR_PIN = 9;
const int STEPPER_STEP_PIN = 10;

// --- Servo Motors ---
Servo servo1, servo2, servo3, servo4;

// --- Modes ---
enum Mode { IDLE, CONTINUOUS, TRIGGER, SCAN_PHASE };
Mode currentMode = IDLE;

// --- Stepper Control ---
unsigned long stepInterval = 500;   // microseconds per step (fixed)
unsigned long lastStepTime = 0;
bool stepState = false;

// --- IR Sensor ---
int lastStableIrState = HIGH;
int lastFlickerIrState = HIGH;
unsigned long lastStateChangeTime = 0;
const long debounceDelay = 50;
unsigned long beamBrokenStartTime = 0;
bool beamIsBroken = false;

// --- Conveyor Parameters ---
const float CONVEYOR_SPEED_IN_PER_SEC = 1.2395; // Calibrated real speed (in/s)
const float SCAN_SEGMENTS[] = {1.0, 7.0, 7.0, 4.5}; // inch distances per scan phase
const int NUM_SEGMENTS = sizeof(SCAN_SEGMENTS) / sizeof(SCAN_SEGMENTS[0]);
const unsigned long STOP_DURATION_MS = 3000; // pause time between scans (3 seconds)

// --- Scan State ---
int currentSegment = 0;
int pauseCount = 0;
bool scanInProgress = false;
bool waitingForPause = false;
bool waitingForBeam = false;
unsigned long moveStartTime = 0;
unsigned long pauseStartTime = 0;
unsigned long segmentDurationMs = 0;
bool stepStateScan = false;
unsigned long lastStepTimeScan = 0;

void setup() {
  Serial.begin(9600);

  pinMode(IR_SENSOR_PIN, INPUT_PULLUP);
  pinMode(STEPPER_ENA_PIN, OUTPUT);
  pinMode(STEPPER_DIR_PIN, OUTPUT);
  pinMode(STEPPER_STEP_PIN, OUTPUT);
  digitalWrite(STEPPER_DIR_PIN, HIGH);
  digitalWrite(STEPPER_ENA_PIN, HIGH);

  servo1.attach(SERVO_1_PIN);
  servo2.attach(SERVO_2_PIN);
  servo3.attach(SERVO_3_PIN);
  servo4.attach(SERVO_4_PIN);
  servo1.write(90);
  servo2.write(90);
  servo3.write(90);
  servo4.write(90);

  Serial.println("Master Controller Initialized. Mode: IDLE");
}

void loop() {
  if (currentMode == SCAN_PHASE)
    handleScanPhase();
  else
    handleStepper();

  checkIrSensor();
  checkSerialCommands();
}

// ---------------- Stepper for Continuous / Trigger ----------------
void handleStepper() {
  if (currentMode == SCAN_PHASE) return;
  bool active = (currentMode == CONTINUOUS || currentMode == TRIGGER);
  digitalWrite(STEPPER_ENA_PIN, active ? LOW : HIGH);
  if (active) {
    unsigned long t = micros();
    if (t - lastStepTime >= stepInterval) {
      lastStepTime = t;
      stepState = !stepState;
      digitalWrite(STEPPER_STEP_PIN, stepState);
    }
  }
}

// ---------------- Scan Phase ----------------
void handleScanPhase() {
  digitalWrite(STEPPER_ENA_PIN, LOW);

  // Conveyor runs while waiting for wood
  if (waitingForBeam && !scanInProgress) {
    unsigned long t = micros();
    if (t - lastStepTimeScan >= stepInterval) {
      lastStepTimeScan = t;
      stepStateScan = !stepStateScan;
      digitalWrite(STEPPER_STEP_PIN, stepStateScan);
    }
    return;
  }

  if (!scanInProgress) return;

  // Segmented scanning logic
  if (currentSegment < NUM_SEGMENTS) {
    if (!waitingForPause) {
      if (moveStartTime == 0) {
        segmentDurationMs = (SCAN_SEGMENTS[currentSegment] / CONVEYOR_SPEED_IN_PER_SEC) * 1000;
        moveStartTime = millis();
        Serial.print("Segment ");
        Serial.print(currentSegment + 1);
        Serial.print(" moving for ");
        Serial.print(segmentDurationMs / 1000.0);
        Serial.println(" s...");
      }

      unsigned long t = micros();
      if (t - lastStepTimeScan >= stepInterval) {
        lastStepTimeScan = t;
        stepStateScan = !stepStateScan;
        digitalWrite(STEPPER_STEP_PIN, stepStateScan);
      }

      if (millis() - moveStartTime >= segmentDurationMs) {
        moveStartTime = 0;
        if (currentSegment < NUM_SEGMENTS - 1) {
          waitingForPause = true;
          pauseStartTime = millis();
          pauseCount++;
          Serial.print("Segment ");
          Serial.print(currentSegment + 1);
          Serial.println(" complete. Pausing...");
        } else {
          // Final scan segment complete
          scanInProgress = false;
          clearTailAfterScan();
        }
      }
    } else if (millis() - pauseStartTime >= STOP_DURATION_MS) {
      waitingForPause = false;
      currentSegment++;
      Serial.print("Resuming segment ");
      Serial.println(currentSegment + 1);
    }
  }
}

// ---------------- Tail Clear After Final Scan ----------------
void clearTailAfterScan() {
  Serial.println("Last scan phase complete. Clearing tail...");

  float clearDistanceInches = 2.0; // move 2 inches more to clear beam
  unsigned long clearDurationMs = (clearDistanceInches / CONVEYOR_SPEED_IN_PER_SEC) * 1000;

  unsigned long clearStart = millis();
  while (millis() - clearStart < clearDurationMs) {
    unsigned long t = micros();
    if (t - lastStepTimeScan >= stepInterval) {
      lastStepTimeScan = t;
      stepStateScan = !stepStateScan;
      digitalWrite(STEPPER_STEP_PIN, stepStateScan);
    }
  }

  // Stop conveyor before waiting for next wood
  digitalWrite(STEPPER_ENA_PIN, HIGH);
  delay(500); // small pause for beam to stabilize

  Serial.println("Tail cleared. Waiting for next beam...");
  currentSegment = 0;
  pauseCount = 0;
  scanInProgress = false;
  waitingForBeam = true;

  // Re-enable motor driver
  digitalWrite(STEPPER_ENA_PIN, LOW);
}

// ---------------- IR Sensor & Measurement ----------------
void checkIrSensor() {
  int currentIrState = digitalRead(IR_SENSOR_PIN);
  if (currentIrState != lastFlickerIrState)
    lastStateChangeTime = millis();
  lastFlickerIrState = currentIrState;

  if ((millis() - lastStateChangeTime) > debounceDelay) {
    if (currentIrState != lastStableIrState) {
      if (currentIrState == LOW) { // Beam broken
        Serial.println("B");
        beamBrokenStartTime = millis();
        beamIsBroken = true;

        if (currentMode == SCAN_PHASE && waitingForBeam) {
          waitingForBeam = false;
          scanInProgress = true;
          currentSegment = 0;
          pauseCount = 0;
          Serial.println("Beam detected! Starting segmented scan...");
        }
      } 
      else if (beamIsBroken) { // Beam cleared
        unsigned long rawDuration = millis() - beamBrokenStartTime;
        beamIsBroken = false;

        // Adjust for scanning pauses
        unsigned long adjustedDuration = (rawDuration > pauseCount * STOP_DURATION_MS)
                                          ? rawDuration - (pauseCount * STOP_DURATION_MS)
                                          : rawDuration;

        float measuredLength = (adjustedDuration * CONVEYOR_SPEED_IN_PER_SEC) / 1000.0;

        Serial.print("L:");
        Serial.print(rawDuration);
        Serial.print(" ms (Adjusted: ");
        Serial.print(adjustedDuration);
        Serial.print(" ms) | Length: ");
        Serial.print(measuredLength, 2);
        Serial.println(" in");

        Serial.println("READY_FOR_NEXT_SCAN");
      }
      lastStableIrState = currentIrState;
    }
  }
}

// ---------------- Serial Commands ----------------
void checkSerialCommands() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    Serial.println(cmd);
    switch (cmd) {
      case '1': activateAllServoGates(90); break;
      case '2': activateAllServoGates(45); break;
      case '3': activateAllServoGates(135); break;
      case '0': activateAllServoGates(0); break;
      case 'C': currentMode = CONTINUOUS; break;
      case 'T': currentMode = TRIGGER; break;
      case 'S':
        currentMode = SCAN_PHASE;
        scanInProgress = false;
        waitingForBeam = true;
        currentSegment = 0;
        Serial.println("Mode: SCAN_PHASE (conveyor running, waiting for beam)");
        break;
      case 'X':
        currentMode = IDLE;
        activateAllServoGates(90);
        scanInProgress = false;
        waitingForBeam = false;
        digitalWrite(STEPPER_ENA_PIN, HIGH);
        Serial.println("Mode: IDLE");
        break;
    }
  }
}

// ---------------- Servo Control ----------------
void activateAllServoGates(int angle) {
  servo1.write(angle);
  servo2.write(angle);
  servo3.write(angle);
  servo4.write(angle);
}
