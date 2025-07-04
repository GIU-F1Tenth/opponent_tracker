from setuptools import find_packages, setup

package_name = 'opponent_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ["*.yaml"]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fam',
    maintainer_email='fam@awadlouis.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "opponent_tracker_node = opponent_tracker.opponent_tracker_node:main",
        ],
    },
)
