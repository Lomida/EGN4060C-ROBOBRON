#include <Arduino.h>
#include <AccelStepper.h>
#include <ESP32Servo.h>
#include "esp_camera.h"

// ---- Camera pins  ----
#define CAMERA_MODEL_ESP32S3_EYE
#include "camera_pins.h"

// ---- Other pins  ----
#define SERVO1_PIN 2  // Arc Adjustment
#define SERVO2_PIN 1  // Latch
#define STEP_A 19
#define DIR_A  20
#define STEP_B 21
#define DIR_B  14

// ---- Objects ----
AccelStepper mA(AccelStepper::DRIVER, STEP_A, DIR_A);
AccelStepper mB(AccelStepper::DRIVER, STEP_B, DIR_B);
Servo s1;
Servo s2;

// ---- Servo params for Hitec HS485HB ----
// 553-2425us for ~190 degrees
static int servoFreqHz  = 50;
static int servoMinUs   = 553;    // Full mechanical minimum
static int servoMaxUs   = 2425;   // Full mechanical maximum

static int servoClosed  = 1100;   // Latch closed position
static int servoOpen    = 1900;   // Latch open/released
static int servoSafeMin = 600;    // Safe operational minimum
static int servoSafeMax = 2400;   // Safe operational maximum
static int rampStepUs   = 8;      // Smooth movement increment

static uint16_t rampEveryMs = 4;  // Ramp update rate
static uint16_t holdMs  = 400;    // Hold time during unlatch sequence

// ---- Debug Mode ----
static bool debugMode = false;
static uint32_t lastDebugPrint = 0;
static uint32_t debugPrintInterval = 1000;

// ---- Coordinated Movement State ----
enum CoordMoveState { COORD_IDLE, COORD_LOADING, COORD_FIRING };
static CoordMoveState coordState = COORD_IDLE;
static uint32_t coordNextUpdate = 0;
static int coordPhase = 0;

// ---- Test Sequence State ----
enum TestState { TEST_IDLE, TEST_SERVO1_FULL, TEST_SERVO2_FULL, TEST_BOTH_MIRROR, TEST_BOTH_OPPOSITE, TEST_HOLD_SERVO1, TEST_HOLD_SERVO2 };
static TestState testState = TEST_IDLE;
static int testPhase = 0;
static uint32_t testNextUpdate = 0;
static int servo1HoldAngle = 0;
static int servo2HoldAngle = 0;
enum ServoState { SERVO_IDLE, SERVO_OPENING, SERVO_HOLD, SERVO_CLOSING, SERVO_GOTO };

// ---- Servo 1 state (Arc Adjustment Servo - on GPIO 2) ----
static ServoState s1State = SERVO_IDLE;
static int s1TargetUs = servoClosed;
static int s1CurrentUs = servoClosed;
static uint32_t s1NextTick = 0, s1HoldUntil = 0;

// ---- Servo 2 state (Latch Servo - on GPIO 1) ----
static ServoState s2State = SERVO_IDLE;
static int s2TargetUs = servoClosed;
static int s2CurrentUs = servoClosed;
static uint32_t s2NextTick = 0, s2HoldUntil = 0;

// ---- Motion Recording for Playback ----
struct MotionPoint {
  long motorA;
  long motorB;
  int servo1Us;
  int servo2Us;
  uint32_t delayMs;
};
static MotionPoint recordedMotion[20];
static int recordCount = 0;
static bool recording = false;
static bool playingBack = false;
static int playbackIndex = 0;
static uint32_t playbackNextTime = 0;

// ---- Streaming protocol ----
static volatile bool streaming = false;
static volatile bool streamingRequested = false;
static const uint32_t STREAM_BAUD = 2000000;
static const uint32_t FRAME_INTERVAL_MS = 90;


// ---- Safe Serial Output ----
static void safePrintln(const String& msg) {
  if (!streaming) {
    Serial.println(msg);
  }
}

static void safePrint(const String& msg) {
  if (!streaming) {
    Serial.print(msg);
  }
}

// ---- Servo helpers ----
static inline void servo1WriteSafeUs(int us) {
  us = constrain(us, servoSafeMin, servoSafeMax);
  s1.writeMicroseconds(us);
  s1CurrentUs = us;
}

static inline void servo2WriteSafeUs(int us) {
  us = constrain(us, servoSafeMin, servoSafeMax);
  s2.writeMicroseconds(us);
  s2CurrentUs = us;
}

static void servo1StartGotoUs(int targetUs) {
  s1TargetUs = constrain(targetUs, servoSafeMin, servoSafeMax);
  s1State = SERVO_GOTO;
  s1NextTick = millis();
}

static void servo2StartGotoUs(int targetUs) {
  s2TargetUs = constrain(targetUs, servoSafeMin, servoSafeMax);
  s2State = SERVO_GOTO;
  s2NextTick = millis();
}

static void servo1StartUnlatch() {
  s1State = SERVO_OPENING;
  s1TargetUs = servoOpen;
  s1NextTick = millis();
}

static void servo2StartUnlatch() {
  s2State = SERVO_OPENING;
  s2TargetUs = servoOpen;
  s2NextTick = millis();
}

static void servo1Idle() {
  s1.detach();
  s1State = SERVO_IDLE;
  safePrintln("Servo 1 power OFF (idle)");
}

static void servo1Reattach() {
  s1.attach(SERVO1_PIN, servoMinUs, servoMaxUs);
  servo1WriteSafeUs(s1CurrentUs);
  safePrintln("Servo 1 power ON (reattached)");
}

static void servo2Idle() {
  s2.detach();
  s2State = SERVO_IDLE;
  safePrintln("Servo 2 power OFF (idle)");
}

static void servo2Reattach() {
  s2.attach(SERVO2_PIN, servoMinUs, servoMaxUs);
  servo1WriteSafeUs(s2CurrentUs);
  safePrintln("Servo 2 power ON (reattached)");
}


// ---- Debug Functions ----
static void printDebugInfo() {
  if (!debugMode || streaming) return;
  uint32_t now = millis();
  if (now - lastDebugPrint < debugPrintInterval) return;
  lastDebugPrint = now;
  
  Serial.print("MA:"); Serial.print(mA.currentPosition());
  Serial.print(" tgt:"); Serial.print(mA.targetPosition());
  Serial.print(" spd:"); Serial.println(mA.speed());
  
  Serial.print("MB:"); Serial.print(mB.currentPosition());
  Serial.print(" tgt:"); Serial.print(mB.targetPosition());
  Serial.print(" spd:"); Serial.println(mB.speed());
  
  Serial.print("S1:"); Serial.print(s1CurrentUs);
  Serial.print("us ("); Serial.print(map(s1CurrentUs, servoMinUs, servoMaxUs, 0, 190));
  Serial.print("°) tgt:"); Serial.print(s1TargetUs);
  Serial.print(" state:"); Serial.println(s1State);
  
  Serial.print("S2:"); Serial.print(s2CurrentUs);
  Serial.print("us ("); Serial.print(map(s2CurrentUs, servoMinUs, servoMaxUs, 0, 190));
  Serial.print("°) tgt:"); Serial.print(s2TargetUs);
  Serial.print(" state:"); Serial.println(s2State);
  
  if (coordState != COORD_IDLE) {
    Serial.print("COORD: state="); Serial.print(coordState);
    Serial.print(" phase="); Serial.println(coordPhase);
  }
  
  if (testState != TEST_IDLE) {
    Serial.print("TEST: state="); Serial.print(testState);
    Serial.print(" phase="); Serial.println(testPhase);
  }
  
}

// ---- Coordinated Movement Functions ----
static void startCoordinatedLoad() {
  coordState = COORD_LOADING;
  coordPhase = 0;
  coordNextUpdate = millis();
  safePrintln("Starting coordinated LOAD sequence...");
}

static void updateCoordinatedMovement() {
  if (coordState == COORD_IDLE) return;
  
  uint32_t now = millis();
  if (now < coordNextUpdate) return;
  
  switch (coordState) {
    case COORD_LOADING:
      switch (coordPhase) {
        case 0:
          safePrintln("LOAD Phase 1: Pulling down...");
          mB.move(-3000);
          servo1StartGotoUs(2000);
          coordPhase++;
          coordNextUpdate = now + 100;
          break;
          
        case 1:
          if (abs(mB.distanceToGo()) < 1500) {
            safePrintln("LOAD Phase 2: Positioning latch...");
            servo2StartGotoUs(1500);
            coordPhase++;
          }
          coordNextUpdate = now + 50;
          break;
          
        case 2:
          if (mB.distanceToGo() == 0) {
            safePrintln("LOAD Phase 3: Securing latch...");
            servo2StartGotoUs(servoClosed);
            coordPhase++;
            coordNextUpdate = now + 500;
          } else {
            coordNextUpdate = now + 50;
          }
          break;
          
        case 3:
          safePrintln("LOAD Phase 4: Retracting...");
          mB.move(1500);
          servo1StartGotoUs(1200);
          coordPhase++;
          coordNextUpdate = now + 100;
          break;
          
        case 4:
          if (mB.distanceToGo() == 0 && s1State == SERVO_IDLE) {
            safePrintln("LOAD Complete! Ready to fire.");
            coordState = COORD_IDLE;
            coordPhase = 0;
          }
          coordNextUpdate = now + 50;
          break;
      }
      break;
      
    default:
      break;
  }
}

// ---- Test Sequence Functions ----
static void startTestSequence(TestState newTest) {
  testState = newTest;
  testPhase = 0;
  testNextUpdate = millis();
  
  switch (newTest) {
    case TEST_SERVO1_FULL:
      safePrintln("TEST: Servo 1 full 190° sweep");
      break;
    case TEST_SERVO2_FULL:
      safePrintln("TEST: Servo 2 full 190° sweep");
      break;
    case TEST_BOTH_MIRROR:
      safePrintln("TEST: Both servos mirror movement");
      break;
    case TEST_BOTH_OPPOSITE:
      safePrintln("TEST: Both servos opposite movement");
      break;
    default:
      break;
  }
}

static void updateTestSequence() {
  if (testState == TEST_IDLE) return;
  
  uint32_t now = millis();
  if (now < testNextUpdate) return;
  
  switch (testState) {
    case TEST_SERVO1_FULL:
      switch (testPhase) {
        case 0: servo1StartGotoUs(servoMinUs); testPhase++; testNextUpdate = now + 2000; break;
        case 1: servo1StartGotoUs(servoMinUs + (servoMaxUs - servoMinUs) / 4); testPhase++; testNextUpdate = now + 1500; break;
        case 2: servo1StartGotoUs((servoMinUs + servoMaxUs) / 2); testPhase++; testNextUpdate = now + 1500; break;
        case 3: servo1StartGotoUs(servoMinUs + 3 * (servoMaxUs - servoMinUs) / 4); testPhase++; testNextUpdate = now + 1500; break;
        case 4: servo1StartGotoUs(servoMaxUs); testPhase++; testNextUpdate = now + 2000; break;
        case 5: servo1StartGotoUs(1500); testPhase++; testNextUpdate = now + 2000; break;
        case 6: 
          safePrintln("Servo 1 test complete!");
          testState = TEST_IDLE;
          break;
      }
      break;
      
    case TEST_SERVO2_FULL:
      switch (testPhase) {
        case 0: servo2StartGotoUs(servoMinUs); testPhase++; testNextUpdate = now + 2000; break;
        case 1: servo2StartGotoUs(servoMinUs + (servoMaxUs - servoMinUs) / 4); testPhase++; testNextUpdate = now + 1500; break;
        case 2: servo2StartGotoUs((servoMinUs + servoMaxUs) / 2); testPhase++; testNextUpdate = now + 1500; break;
        case 3: servo2StartGotoUs(servoMinUs + 3 * (servoMaxUs - servoMinUs) / 4); testPhase++; testNextUpdate = now + 1500; break;
        case 4: servo2StartGotoUs(servoMaxUs); testPhase++; testNextUpdate = now + 2000; break;
        case 5: servo2StartGotoUs(1500); testPhase++; testNextUpdate = now + 2000; break;
        case 6: 
          safePrintln("Servo 2 test complete!");
          testState = TEST_IDLE;
          break;
      }
      break;
      
    case TEST_BOTH_MIRROR:
      switch (testPhase) {
        case 0: 
          servo1StartGotoUs(servoMinUs);
          servo2StartGotoUs(servoMinUs);
          testPhase++;
          testNextUpdate = now + 2000;
          break;
        case 1:
          servo1StartGotoUs(servoMaxUs);
          servo2StartGotoUs(servoMaxUs);
          testPhase++;
          testNextUpdate = now + 3000;
          break;
        case 2:
          servo1StartGotoUs(1500);
          servo2StartGotoUs(1500);
          testPhase++;
          testNextUpdate = now + 2000;
          break;
        case 3:
          safePrintln("Mirror test complete!");
          testState = TEST_IDLE;
          break;
      }
      break;
      
    case TEST_BOTH_OPPOSITE:
      switch (testPhase) {
        case 0:
          servo1StartGotoUs(servoMinUs);
          servo2StartGotoUs(servoMaxUs);
          testPhase++;
          testNextUpdate = now + 2000;
          break;
        case 1:
          servo1StartGotoUs(servoMaxUs);
          servo2StartGotoUs(servoMinUs);
          testPhase++;
          testNextUpdate = now + 3000;
          break;
        case 2:
          servo1StartGotoUs(1500);
          servo2StartGotoUs(1500);
          testPhase++;
          testNextUpdate = now + 2000;
          break;
        case 3:
          safePrintln("Opposite test complete!");
          testState = TEST_IDLE;
          break;
      }
      break;

    case TEST_HOLD_SERVO1:
    if (testPhase == 0) {
      int holdUs = map(servo1HoldAngle, 0, 190, servoMinUs, servoMaxUs);
      servo1StartGotoUs(holdUs);
      safePrint("Holding Servo 1 at ");
      safePrint(String(servo1HoldAngle));
      safePrintln("°");
      testPhase++;
    }
    break;

    case TEST_HOLD_SERVO2:
    if (testPhase == 0) {
      int holdUs = map(servo2HoldAngle, 0, 190, servoMinUs, servoMaxUs);
      servo2StartGotoUs(holdUs);
      safePrint("Holding Servo 2 at ");
      safePrint(String(servo2HoldAngle));
      safePrintln("°");
      testPhase++;
    }
    break;
    
    default:
      testState = TEST_IDLE;
      break;
  }
}

static void servo1Update() {
  uint32_t now = millis();
  if (now < s1NextTick) return;

  switch (s1State) {
    case SERVO_IDLE: return;

    case SERVO_GOTO: {
      if (abs(s1CurrentUs - s1TargetUs) <= rampStepUs) {
        servo1WriteSafeUs(s1TargetUs);
        s1State = SERVO_IDLE;
        return;
      }
      int dir = (s1TargetUs > s1CurrentUs) ? +1 : -1;
      int step = (abs(s1TargetUs - s1CurrentUs) > 100) ? rampStepUs * 2 : rampStepUs;
      servo1WriteSafeUs(s1CurrentUs + dir * step);
      s1NextTick = now + rampEveryMs;
      break;
    }

    case SERVO_OPENING: {
      if (s1CurrentUs >= s1TargetUs) {
        s1State = SERVO_HOLD;
        s1HoldUntil = now + holdMs;
      } else {
        servo1WriteSafeUs(s1CurrentUs + rampStepUs);
        s1NextTick = now + rampEveryMs;
      }
      break;
    }

    case SERVO_HOLD: {
      if (now >= s1HoldUntil) {
        s1State = SERVO_CLOSING;
        s1TargetUs = servoClosed;
      }
      s1NextTick = now + rampEveryMs;
      break;
    }

    case SERVO_CLOSING: {
      if (s1CurrentUs <= s1TargetUs) {
        s1State = SERVO_IDLE;
      } else {
        servo1WriteSafeUs(s1CurrentUs - rampStepUs);
        s1NextTick = now + rampEveryMs;
      }
      break;
    }
  }
}

static void servo2Update() {
  uint32_t now = millis();
  if (now < s2NextTick) return;

  switch (s2State) {
    case SERVO_IDLE: return;

    case SERVO_GOTO: {
      if (abs(s2CurrentUs - s2TargetUs) <= rampStepUs) {
        servo2WriteSafeUs(s2TargetUs);
        s2State = SERVO_IDLE;
        return;
      }
      int dir = (s2TargetUs > s2CurrentUs) ? +1 : -1;
      int step = (abs(s2TargetUs - s2CurrentUs) > 100) ? rampStepUs * 2 : rampStepUs;
      servo2WriteSafeUs(s2CurrentUs + dir * step);
      s2NextTick = now + rampEveryMs;
      break;
    }

    case SERVO_OPENING: {
      if (s2CurrentUs >= s2TargetUs) {
        s2State = SERVO_HOLD;
        s2HoldUntil = now + holdMs;
      } else {
        servo2WriteSafeUs(s2CurrentUs + rampStepUs);
        s2NextTick = now + rampEveryMs;
      }
      break;
    }

    case SERVO_HOLD: {
      if (now >= s2HoldUntil) {
        s2State = SERVO_CLOSING;
        s2TargetUs = servoClosed;
      }
      s2NextTick = now + rampEveryMs;
      break;
    }

    case SERVO_CLOSING: {
      if (s2CurrentUs <= s2TargetUs) {
        s2State = SERVO_IDLE;
      } else {
        servo2WriteSafeUs(s2CurrentUs - rampStepUs);
        s2NextTick = now + rampEveryMs;
      }
      break;
    }
  }
}

static inline void writeLE16(uint16_t v) {
  uint8_t b[2] = { uint8_t(v), uint8_t(v >> 8) };
  Serial.write(b, 2);
}

static inline void writeLE32(uint32_t v) {
  uint8_t b[4] = { uint8_t(v), uint8_t(v >> 8), uint8_t(v >> 16), uint8_t(v >> 24) };
  Serial.write(b, 4);
}

// ---- Tasks ----
TaskHandle_t hStreamTask = nullptr;
TaskHandle_t hMotionTask = nullptr;

// Motion task: steppers + servos (Core 1)
void motionTask(void*) {
  for (;;) {
    mA.run();
    mB.run();
    servo1Update();
    servo2Update();
    updateCoordinatedMovement();
    updateTestSequence();
    vTaskDelay(1);
  }
}

// Camera/stream task (Core 0)
void streamTask(void*) {
  uint32_t lastFrameMs = millis();
  
  for (;;) {
    // Check if we should start/stop streaming
    if (streamingRequested && !streaming) {
      streaming = true;
      lastFrameMs = millis();
    } else if (!streamingRequested && streaming) {
      streaming = false;
    }
    
    if (streaming) {
      uint32_t now = millis();
      if (now - lastFrameMs >= FRAME_INTERVAL_MS) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) {
          if (fb->format == PIXFORMAT_JPEG) {
            const char magic[4] = { 'M','J','P','G' };
            Serial.write((const uint8_t*)magic, 4);
            writeLE32(fb->len);
            writeLE16(fb->width);
            writeLE16(fb->height);

            const uint8_t* p = fb->buf;
            size_t left = fb->len;
            const size_t CHUNK = 256;
            while (left) {
              size_t n = (left > CHUNK) ? CHUNK : left;
              Serial.write(p, n);
              p += n;
              left -= n;
              taskYIELD();
            }
          }
          esp_camera_fb_return(fb);
        }
        lastFrameMs = now;
      }
    }
    vTaskDelay(1);
  }
}

// ---- Line commands from host ----
static String lineBuf;

static void parseInputLine(const String& line) {
  String s = line;
  s.trim();
  s.toUpperCase();
  if (!s.length()) return;

  // ----- Debug Commands -----
  if (s == "DEBUG") {
    debugMode = !debugMode;
    safePrint("Debug mode: ");
    safePrintln(debugMode ? "ON" : "OFF");
    return;
  }

  // ----- Full Range Test Commands -----
  if (s == "FULLTEST1") {
    startTestSequence(TEST_SERVO1_FULL);
    return;
  }
  
  if (s == "FULLTEST2") {
    startTestSequence(TEST_SERVO2_FULL);
    return;
  }
  
  if (s == "TESTMIRROR") {
    startTestSequence(TEST_BOTH_MIRROR);
    return;
  }
  
  if (s == "TESTOPP") {
    startTestSequence(TEST_BOTH_OPPOSITE);
    return;
  }

  // ----- Coordinated Movement Commands -----
  if (s == "COORDLOAD" || s == "CL") {
    startCoordinatedLoad();
    return;
  }

  // ----- Motion Recording Commands -----
  if (s == "RECORD") {
    recording = true;
    recordCount = 0;
    safePrintln("Recording motion... (max 20 points)");
    return;
  }
  
  if (s == "STOPRECORD") {
    recording = false;
    safePrint("Recorded ");
    safePrint(String(recordCount));
    safePrintln(" motion points");
    return;
  }
  
  if (s == "PLAYBACK") {
    if (recordCount > 0) {
      playingBack = true;
      playbackIndex = 0;
      playbackNextTime = millis();
      safePrintln("Playing back recorded motion...");
    } else {
      safePrintln("No motion recorded!");
    }
    return;
  }
  
  if (s == "MARK" && recording && recordCount < 20) {
    recordedMotion[recordCount].motorA = mA.currentPosition();
    recordedMotion[recordCount].motorB = mB.currentPosition();
    recordedMotion[recordCount].servo1Us = s1CurrentUs;
    recordedMotion[recordCount].servo2Us = s2CurrentUs;
    recordedMotion[recordCount].delayMs = 1000;
    recordCount++;
    safePrint("Marked position ");
    safePrintln(String(recordCount));
    return;
  }

  // ----- Servo Position Presets -----
  if (s == "S1P0") {
    servo1StartGotoUs(servoMinUs);
    safePrintln("Servo 1 -> 0%");
    return;
  }
  if (s == "S1P25") {
    servo1StartGotoUs(servoMinUs + (servoMaxUs - servoMinUs) / 4);
    safePrintln("Servo 1 -> 25%");
    return;
  }
  if (s == "S1P50") {
    servo1StartGotoUs((servoMinUs + servoMaxUs) / 2);
    safePrintln("Servo 1 -> 50%");
    return;
  }
  if (s == "S1P75") {
    servo1StartGotoUs(servoMinUs + 3 * (servoMaxUs - servoMinUs) / 4);
    safePrintln("Servo 1 -> 75%");
    return;
  }
  if (s == "S1P100") {
    servo1StartGotoUs(servoMaxUs);
    safePrintln("Servo 1 -> 100%");
    return;
  }
  
  // Servo 2 position presets
  if (s == "S2P0") {
    servo2StartGotoUs(servoMinUs);
    safePrintln("Servo 2 -> 0%");
    return;
  }
  if (s == "S2P25") {
    servo2StartGotoUs(servoMinUs + (servoMaxUs - servoMinUs) / 4);
    safePrintln("Servo 2 -> 25%");
    return;
  }
  if (s == "S2P50") {
    servo2StartGotoUs((servoMinUs + servoMaxUs) / 2);
    safePrintln("Servo 2 -> 50%");
    return;
  }
  if (s == "S2P75") {
    servo2StartGotoUs(servoMinUs + 3 * (servoMaxUs - servoMinUs) / 4);
    safePrintln("Servo 2 -> 75%");
    return;
  }
  if (s == "S2P100") {
    servo2StartGotoUs(servoMaxUs);
    safePrintln("Servo 2 -> 100%");
    return;
  }

  // Quick position tests
  if (s == "S1MIN") {
    servo1StartGotoUs(servoMinUs);
    safePrint("Servo 1 -> MIN (");
    safePrint(String(servoMinUs));
    safePrintln("us)");
    return;
  }
  
  if (s == "S1CENTER" || s == "S1C") {
    servo1StartGotoUs(1500);
    safePrintln("Servo 1 -> CENTER");
    return;
  }
  
  if (s == "S1MAX") {
    servo1StartGotoUs(servoMaxUs);
    safePrint("Servo 1 -> MAX (");
    safePrint(String(servoMaxUs));
    safePrintln("us)");
    return;
  }

  if (s == "S2MIN") {
    servo2StartGotoUs(servoMinUs);
    safePrint("Servo 2 -> MIN (");
    safePrint(String(servoMinUs));
    safePrintln("us)");
    return;
  }
  
  if (s == "S2CENTER" || s == "S2C") {
    servo2StartGotoUs(1500);
    safePrintln("Servo 2 -> CENTER");
    return;
  }
  
  if (s == "S2MAX") {
    servo2StartGotoUs(servoMaxUs);
    safePrint("Servo 2 -> MAX (");
    safePrint(String(servoMaxUs));
    safePrintln("us)");
    return;
  }

  // ----- Status/Info Commands -----
  if (s == "STATUS" || s == "?") {
    safePrint("MA:");
    safePrint(String(mA.currentPosition()));
    safePrint(" MB:");
    safePrint(String(mB.currentPosition()));
    safePrint(" S1:");
    safePrint(String(map(s1CurrentUs, servoMinUs, servoMaxUs, 0, 190)));
    safePrint("° S2:");
    safePrint(String(map(s2CurrentUs, servoMinUs, servoMaxUs, 0, 190)));
    safePrintln("°");
    return;
  }

  if (s == "INFO") {
    safePrintln("=== Catapult Turret Status ===");
    safePrint("Motor A: ");
    safePrintln(String(mA.currentPosition()));
    safePrint("Motor B: ");
    safePrintln(String(mB.currentPosition()));
    safePrint("Servo 1: ");
    safePrint(String(s1CurrentUs));
    safePrint("us (");
    safePrint(String(map(s1CurrentUs, servoMinUs, servoMaxUs, 0, 190)));
    safePrintln("°)");
    safePrint("Servo 2: ");
    safePrint(String(s2CurrentUs));
    safePrint("us (");
    safePrint(String(map(s2CurrentUs, servoMinUs, servoMaxUs, 0, 190)));
    safePrintln("°)");
    return;
  }

  // ----- Servo 2 commands (Latch) -----
  if (s == "U2") { 
    servo2StartUnlatch();
    safePrintln("Servo 2 unlatch");
    return;
  }
  
  if (s.startsWith("G2 ")) {
    int deg = constrain(s.substring(3).toInt(), 0, 190);
    int us  = map(deg, 0, 190, servoMinUs, servoMaxUs);
    servo2StartGotoUs(us);
    return;
  }
  
  if (s.startsWith("US2 ")) {
    int us = constrain(s.substring(4).toInt(), servoSafeMin, servoSafeMax);
    servo2StartGotoUs(us);
    return;
  }

  if (s.startsWith("R2 ")) {
    int degDelta = s.substring(3).toInt();
    int currentDeg = map(s2CurrentUs, servoMinUs, servoMaxUs, 0, 190);
    int newDeg = constrain(currentDeg + degDelta, 0, 190);
    int newUs = map(newDeg, 0, 190, servoMinUs, servoMaxUs);
    servo2StartGotoUs(newUs);
    return;
  }

  // ----- Servo 1 (Arc) commands -----
  if (s == "U") { 
    servo1StartUnlatch();
    safePrintln("Servo 1 unlatch");
    return;
  }
  
  if (s.startsWith("G ")) {
    int deg = constrain(s.substring(2).toInt(), 0, 190);
    int us  = map(deg, 0, 190, servoMinUs, servoMaxUs);
    servo1StartGotoUs(us);
    return;
  }
  
  if (s.startsWith("US ")) {
    int us = constrain(s.substring(3).toInt(), servoSafeMin, servoSafeMax);
    servo1StartGotoUs(us);
    return;
  }

  if (s.startsWith("R ")) {
    int degDelta = s.substring(2).toInt();
    int currentDeg = map(s1CurrentUs, servoMinUs, servoMaxUs, 0, 190);
    int newDeg = constrain(currentDeg + degDelta, 0, 190);
    int newUs = map(newDeg, 0, 190, servoMinUs, servoMaxUs);
    servo1StartGotoUs(newUs);
    return;
  }

  // ----- High-Level Sequence Commands -----
  if (s == "FIRE") {
    safePrintln("FIRING!");
    servo2StartUnlatch();
    return;
  }

  if (s == "LOAD") {
    safePrintln("Loading catapult...");
    mB.move(-3000);
    servo1StartGotoUs(servoOpen);
    return;
  }

  if (s == "RESET") {
    safePrintln("Resetting to safe position...");
    servo1StartGotoUs(servoClosed);
    servo2StartGotoUs(servoClosed);
    mA.moveTo(0);
    mB.moveTo(0);
    coordState = COORD_IDLE;
    testState = TEST_IDLE;
    return;
  }

  if (s == "STOP") {
    safePrintln("Emergency stop!");
    mA.stop();
    mB.stop();
    coordState = COORD_IDLE;
    testState = TEST_IDLE;
    return;
  }

  // ----- Stepper Commands -----
  if (s.startsWith("A ")) {
    long n = s.substring(2).toInt();
    mA.move(n);
    return;
  }
  
  if (s.startsWith("B ")) {
    long n = s.substring(2).toInt();
    mB.move(n);
    return;
  }

  if (s.startsWith("MA ")) {
    long pos = s.substring(3).toInt();
    mA.moveTo(pos);
    return;
  }
  
  if (s.startsWith("MB ")) {
    long pos = s.substring(3).toInt();
    mB.moveTo(pos);
    return;
  }

  // ----- Speed/Acceleration -----
  if (s.startsWith("V ")) {
    float v = s.substring(2).toFloat();
    mA.setMaxSpeed(v);
    mB.setMaxSpeed(v);
    safePrint("Speed: ");
    safePrintln(String(v));
    return;
  }
  
  if (s.startsWith("ACC ")) {
    float a = s.substring(4).toFloat();
    mA.setAcceleration(a);
    mB.setAcceleration(a);
    safePrint("Accel: ");
    safePrintln(String(a));
    return;
  }

  // Zero positions
  if (s == "ZEROA" || s == "ZA") {
    mA.setCurrentPosition(0);
    safePrintln("Motor A zeroed");
    return;
  }
  
  if (s == "ZEROB" || s == "ZB") {
    mB.setCurrentPosition(0);
    safePrintln("Motor B zeroed");
    return;
  }

  if (s.startsWith("W ")) {
  servo1HoldAngle = constrain(s.substring(2).toInt(), -10, 190);
  startTestSequence(TEST_HOLD_SERVO1);
  return;
  }

  if (s.startsWith("L ")) {
  servo2HoldAngle = constrain(s.substring(2).toInt(), -10, 190);
  startTestSequence(TEST_HOLD_SERVO2);
  return;
  }

  if (s == "E") {
    servo1Idle();
    testState = TEST_IDLE;
    return;
  }

  if (s == "REATTACH1" || s == "R1") {
    servo1Reattach();
    return;
  }

  if (s == "Z") {
    servo2Idle();
    testState = TEST_IDLE;
    return;
  }

  if (s == "REATTACH2" || s == "R2") {
    servo2Reattach();
    return;
  }

  // Help command
  if (s == "HELP" || s == "H") {
    if (!streaming) {

    }
    return;
  }

  // Unknown command
  if (!streaming) {
    Serial.print("Unknown: ");
    Serial.println(s);
  }
}

// ---- Playback Update ----
void updatePlayback() {
  if (!playingBack || playbackIndex >= recordCount) {
    if (playingBack && playbackIndex >= recordCount) {
      safePrintln("Playback complete!");
      playingBack = false;
    }
    return;
  }
  
  uint32_t now = millis();
  if (now >= playbackNextTime) {
    MotionPoint& pt = recordedMotion[playbackIndex];
    mA.moveTo(pt.motorA);
    mB.moveTo(pt.motorB);
    servo1StartGotoUs(pt.servo1Us);
    servo2StartGotoUs(pt.servo2Us);
    
    playbackIndex++;
    if (playbackIndex < recordCount) {
      playbackNextTime = now + recordedMotion[playbackIndex - 1].delayMs;
    }
  }
}

void setup() {
  // Keep pins quiet at boot
  pinMode(STEP_A, OUTPUT); digitalWrite(STEP_A, LOW);
  pinMode(DIR_A,  OUTPUT); digitalWrite(DIR_A,  LOW);
  pinMode(STEP_B, OUTPUT); digitalWrite(STEP_B, LOW);
  pinMode(DIR_B,  OUTPUT); digitalWrite(DIR_B,  LOW);
  delay(150);

  Serial.begin(STREAM_BAUD);
  delay(500);
  
  // Clear serial buffer
  while (Serial.available()) {
    Serial.read();
  }
  
  Serial.println("Ready for commands");

  // Camera config
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer   = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;  c.pin_d1 = Y3_GPIO_NUM;  c.pin_d2 = Y4_GPIO_NUM;  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;  c.pin_d5 = Y7_GPIO_NUM;  c.pin_d6 = Y8_GPIO_NUM;  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM; c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size   = FRAMESIZE_QVGA;
  c.jpeg_quality = 18;
  c.fb_count     = 3;
  c.fb_location  = CAMERA_FB_IN_PSRAM;
  c.grab_mode    = CAMERA_GRAB_LATEST;

  if (esp_camera_init(&c) == ESP_OK) {
    Serial.println("Camera initialized");
    if (auto s0 = esp_camera_sensor_get()) {
      s0->set_vflip(s0, 1);
    }
  } else {
    Serial.println("Camera init failed, something has gone wrong");
  }

  // Steppers
  mA.setMaxSpeed(1800);
  mA.setAcceleration(3500);
  mA.setMinPulseWidth(2);

  mB.setMaxSpeed(1800);
  mB.setAcceleration(3500);
  mB.setMinPulseWidth(2);

  // Servos
  s1.setPeriodHertz(servoFreqHz);
  s1.attach(SERVO1_PIN, servoMinUs, servoMaxUs);
  servo1WriteSafeUs(1500);

  s2.setPeriodHertz(servoFreqHz);
  s2.attach(SERVO2_PIN, servoMinUs, servoMaxUs);
  servo2WriteSafeUs(1500);

  Serial.print("Servo range: ");
  Serial.print(servoMinUs);
  Serial.print("-");
  Serial.print(servoMaxUs);
  Serial.println("us (190°)");

  // Tasks
  xTaskCreatePinnedToCore(streamTask, "stream", 4096, nullptr, 2, &hStreamTask, 0);
  xTaskCreatePinnedToCore(motionTask, "motion", 4096, nullptr, 3, &hMotionTask, 1);

  Serial.println("System ready!");
}

void loop() {
  // Update systems
  updatePlayback();
  printDebugInfo();
  
  // Process serial input
  while (Serial.available()) {
    int ch = Serial.read();
    
    // Special single-char streaming commands
    if (ch == 'S') {
      streamingRequested = true;
      if (!streaming) {
        Serial.println("Stream ON");
      }
      continue;
    }
    
    if (ch == 'P' || ch == 'Q') {
      streamingRequested = false;
      if (streaming) {
        // Wait to stop
        delay(100);
      }
      continue;
    }
    
    // Build command line
    if (ch == '\n' || ch == '\r') {
      if (lineBuf.length()) {
        parseInputLine(lineBuf);
        lineBuf = "";
      }
    } else if (ch >= 32 && ch < 127 && lineBuf.length() < 64) {
      lineBuf += char(ch);
    }
  }
  
  vTaskDelay(1);
}
