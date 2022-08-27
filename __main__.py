import pathlib

import pulumi
import pulumi_aws as aws
import pulumi_awsx as awsx
import pulumi_command as command


# Project configurations
PROJECT_NAME = pulumi.get_project()

config = pulumi.Config()

EC2_IMAGE_NAME = config.get(
    key='ec2_image_name',
    default='Deep Learning AMI GPU PyTorch 1.12.0 (Amazon Linux 2)*',
)
EC2_INSTANCE_TYPE = config.get(
    key='ec2_instance_type',
    default='p3.2xlarge',
)
EC2_USERNAME = config.get(
    key='ec2_username',
    default='ec2-user',
)
PRIVATE_KEY_PATH = pathlib.Path(
    config.get(
        key='public_key_path',
        default=pathlib.Path(pathlib.Path.home(), '.ssh', 'id_rsa'),
    )
)
PUBLIC_KEY_PATH = PRIVATE_KEY_PATH.with_suffix('.pub')
PROJECT_HOME = f'/home/{EC2_USERNAME}/evalRS-CIKM-2022'


# Importing user's keys
assert PUBLIC_KEY_PATH.exists(), f'Public key {PUBLIC_KEY_PATH} does not exist'
assert PRIVATE_KEY_PATH.exists(), f'Public key {PRIVATE_KEY_PATH} does not exist'

PUBLIC_KEY = PUBLIC_KEY_PATH.read_text('utf-8')
PRIVATE_KEY = PRIVATE_KEY_PATH.read_text('utf-8')


# Creating new key pair
ec2_key_pair = aws.ec2.KeyPair(
    resource_name=f'{PROJECT_NAME}-keypair',
    public_key=PUBLIC_KEY,
)


# Creating VPC
vpc = awsx.ec2.Vpc(
    resource_name=f'{PROJECT_NAME}-vpc',
    number_of_availability_zones=1,
    nat_gateways=awsx.ec2.NatGatewayConfigurationArgs(
        strategy=awsx.ec2.NatGatewayStrategy.NONE,
    ),
    subnet_specs=[
        awsx.ec2.SubnetSpecArgs(
            cidr_mask=24,
            type=awsx.ec2.SubnetType.PUBLIC,
        ),
    ],
)


# Searching for the latest AMI image
ami_image = aws.ec2.get_ami(
    most_recent=True,
    filters=[
        aws.ec2.GetAmiFilterArgs(
            name='name',
            values=[EC2_IMAGE_NAME],
        ),
        aws.ec2.GetAmiFilterArgs(
            name='virtualization-type',
            values=['hvm'],
        ),
        aws.ec2.GetAmiFilterArgs(
            name='architecture',
            values=['x86_64'],
        ),
    ],
    owners=['amazon'],
)


# Definition of the security group to allow SSH access
security_group = aws.ec2.SecurityGroup(
    resource_name=f'{PROJECT_NAME}-security-group',
    vpc_id=vpc.vpc_id,
    description='Enable SSH access',
    ingress=[
        aws.ec2.SecurityGroupIngressArgs(
            protocol='tcp',
            from_port=22,
            to_port=22,
            cidr_blocks=['0.0.0.0/0'],
            self=True,
        ),
    ],
    egress=[
        aws.ec2.SecurityGroupEgressArgs(
            protocol='-1',
            from_port=0,
            to_port=0,
            cidr_blocks=['0.0.0.0/0'],
        ),
    ],
)


# Definition of the EC2 instance
ec2_instance = aws.ec2.Instance(
    resource_name=f'{PROJECT_NAME}-ec2',
    ami=ami_image.id,
    vpc_security_group_ids=[security_group.id],
    instance_type=EC2_INSTANCE_TYPE,
    key_name=ec2_key_pair.id,
    subnet_id=vpc.public_subnet_ids[0],
    tags={
        'Name': f'{PROJECT_NAME}-ec2',
    },
)


# Final command to copy the local .env file
ec2_connection = command.remote.ConnectionArgs(
    host=ec2_instance.public_ip,
    user='ec2-user',
    private_key=PRIVATE_KEY,
)

git_clone_cmd = command.remote.Command(
    resource_name='git-clone',
    connection=ec2_connection,
    create='git clone https://github.com/RecList/evalRS-CIKM-2022'
)
copy_dotenv_cmd = command.remote.CopyFile(
    resource_name='.env',
    connection=ec2_connection,
    local_path='./.env',
    remote_path=f'{PROJECT_HOME}/.env',
    opts=pulumi.ResourceOptions(depends_on=[git_clone_cmd]),
)
pip_install_cmd = command.remote.Command(
    resource_name='pip-install',
    connection=ec2_connection,
    create=f'cd {PROJECT_HOME} && pip install -r ./requirements.txt',
    opts=pulumi.ResourceOptions(depends_on=[copy_dotenv_cmd]),
)


pulumi.export('publicIp', ec2_instance.public_ip)
