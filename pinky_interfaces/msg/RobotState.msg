# msg/RobotState.msg
builtin_interfaces/Time stamp

string robot_name
string system_state

string pose_frame
float32 pose_x
float32 pose_y
float32 pose_yaw

float32 vel_vx
float32 vel_vy
float32 vel_wz

bool is_charging
float32 battery_soc
float32 battery_voltage

int32 primary_id
float32 lost_sec
string vision_json

string mission_id
string plan_json

int32 current_index
string queue_status
string history_json

bool roe_ok
bool safe_backstop
float32 max_speed
string last_violation

string events_json
