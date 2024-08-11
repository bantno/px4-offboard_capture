import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, VehicleCommand, VehicleLocalPosition, VehicleStatus
from px4_msgs.msg import TrajectorySetpoint, TrajectoryBezier, VehicleTrajectoryBezier

# from px4_offboard.path_planner_base import PathPlanner
import numpy as np



class Demo(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__('offboard_control_takeoff_and_land')

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self_prefix = '/px4_1'

        target_prefix = '/px4_2'

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, self_prefix+'/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, self_prefix+'/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, self_prefix+'/fmu/in/vehicle_command', qos_profile)
        self.trajectory_bezier_publisher = self.create_publisher(
            VehicleTrajectoryBezier, self_prefix+'/fmu/in/vehicle_trajectory_bezier', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, self_prefix+'/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, self_prefix+'/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        
        # Create subscribers
        self.target_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, target_prefix+'/fmu/out/vehicle_local_position', self.target_local_position_callback, qos_profile)

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.done = False
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.takeoff_height = -1.0
        self.path_is_valid = False
        # self.planner = PathPlanner()
        self.i=0
        
        self.dt = 0.05
        self.plan_time = 5.0

        # Create a timer to publish control commands
        self.timer = self.create_timer(self.dt, self.timer_callback)

        # Create timer to publish heartbeat signal
        self.heartbeat_timer = self.create_timer(0.25, self.heartbeat_callback)

        # Create a timer to plan path
        self.plan_timer = self.create_timer(self.plan_time,self.plan_callback)

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position

    def target_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber for the target drone."""
        self.target_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def plan_callback(self):
        """Callback function for periodic planning"""
        start=[self.vehicle_local_position.x,self.vehicle_local_position.y,self.vehicle_local_position.z]
        goal=[self.target_local_position.x,self.target_local_position.y,self.target_local_position.z]
        self.path = self.plan_path(start, goal)
        self.i=0
        self.path_is_valid=True

    def servo_set(self,pos):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_ACTUATOR, param1=pos)
        self.get_logger().info('Servo command sent')

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 1.57079  # (90 degree)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing position setpoints {[x, y, z]}")

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 1.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def plan_path(self, start, goal):
        # Define the intermediate point (half a meter above the goal in the -z direction)
        intermediate = np.array([goal[0], goal[1], goal[2] - 0.5])

        if np.linalg.norm(np.array(start) - np.array(goal)) <= 1.0 and start[2] < goal[2]-0.5:
            # Go directly to the goal
            path = np.array([start, goal])

        else:
            # Bezier control points: start, intermediate, goal
            control_points = np.array([start, intermediate, goal])        

            # Function to calculate the Bezier point at t
            def bezier_point(t, control_points):
                n = len(control_points) - 1
                point = np.zeros(3)
                for i in range(n + 1):
                    binomial_coeff = np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))
                    point += binomial_coeff * (1 - t)**(n - i) * t**i * control_points[i]
                return point

            # Generate the Bezier curve points
            t_values = np.linspace(0, 1, 100)
            path = np.array([bezier_point(t, control_points) for t in t_values])

        return path


    def heartbeat_callback(self) -> None:
        """Publish offboard control heartbeat signal"""
        self.publish_offboard_control_heartbeat_signal()

    def timer_callback(self) -> None:
        """Callback function for the timer."""
        
        if self.offboard_setpoint_counter == 10:
            self.arm()
            self.engage_offboard_mode()
            
        if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and self.vehicle_local_position.z > self.takeoff_height:
            self.publish_position_setpoint(0.0, 0.0, self.takeoff_height-.1)
            self.servo_set(-1.0)

        
        
        elif self.vehicle_local_position.z<self.takeoff_height and self.path_is_valid:
            x = self.path[self.i][0]
            y = self.path[self.i][1]
            z = self.path[self.i][2]
            self.publish_position_setpoint(x, y, z)
            if self.i<len(self.path) :
                self.i+=1

        elif self.done:
            self.land()
            exit(0)

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1
            


def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = Demo()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)