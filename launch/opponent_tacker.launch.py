from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    opponent_tracker_config = os.path.join(
        get_package_share_directory('opponent_tracker'),
        'config',
        'opponent_tracker_params.yaml'
    )

    opponent_tracker_node = Node(
        package='opponent_tracker',
        executable='opponent_tracker_node',
        name='opponent_tracker_node',
        output='screen',
        parameters=[opponent_tracker_config],
    )

    return LaunchDescription([
        opponent_tracker_node
    ])
