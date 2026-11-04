from setuptools import find_packages, setup

package_name = 'ur3e_imitation_learning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='thijs',
    maintainer_email='thijswelling@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'record_episodes = ur3e_imitation_learning.record_episodes:main',
            'run_inference = ur3e_imitation_learning.run_inference:main',
            'convert_to_lerobot = ur3e_imitation_learning.convert_to_lerobot:main',
        ],
    },
)
