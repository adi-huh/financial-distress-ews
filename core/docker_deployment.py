"""
Day 24: Docker & Deployment Configuration
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json


class ContainerEnvironment(Enum):
    """Container environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DockerConfig:
    """Docker configuration"""
    image_name: str
    image_tag: str
    port: int = 5000
    environment: ContainerEnvironment = ContainerEnvironment.PRODUCTION
    volumes: List[str] = field(default_factory=list)
    environment_vars: Dict[str, str] = field(default_factory=dict)


class DockerfileGenerator:
    """Generate Dockerfile"""
    
    def __init__(self, base_image: str = "python:3.13-slim"):
        self.base_image = base_image
        self.instructions = []
    
    def add_workdir(self, path: str):
        """Add WORKDIR instruction"""
        self.instructions.append(f"WORKDIR {path}")
    
    def add_copy(self, src: str, dest: str):
        """Add COPY instruction"""
        self.instructions.append(f"COPY {src} {dest}")
    
    def add_run(self, command: str):
        """Add RUN instruction"""
        self.instructions.append(f"RUN {command}")
    
    def add_expose(self, port: int):
        """Add EXPOSE instruction"""
        self.instructions.append(f"EXPOSE {port}")
    
    def add_cmd(self, command: str):
        """Add CMD instruction"""
        self.instructions.append(f"CMD {command}")
    
    def generate(self) -> str:
        """Generate Dockerfile content"""
        content = f"FROM {self.base_image}\n\n"
        content += "\n".join(self.instructions)
        return content


class DockerComposeGenerator:
    """Generate docker-compose.yml"""
    
    def __init__(self):
        self.services = {}
        self.networks = {}
        self.volumes = {}
    
    def add_service(self, name: str, config: Dict[str, Any]):
        """Add service"""
        self.services[name] = config
    
    def add_network(self, name: str, driver: str = "bridge"):
        """Add network"""
        self.networks[name] = {"driver": driver}
    
    def add_volume(self, name: str, driver: str = "local"):
        """Add volume"""
        self.volumes[name] = {"driver": driver}
    
    def generate(self) -> Dict[str, Any]:
        """Generate docker-compose structure"""
        return {
            "version": "3.8",
            "services": self.services,
            "networks": self.networks,
            "volumes": self.volumes
        }


class KubernetesDeployment:
    """Kubernetes deployment configuration"""
    
    def __init__(self, app_name: str):
        self.app_name = app_name
        self.replicas = 3
        self.image = ""
        self.resources = {}
    
    def set_image(self, image: str):
        """Set container image"""
        self.image = image
    
    def set_replicas(self, replicas: int):
        """Set pod replicas"""
        self.replicas = replicas
    
    def set_resources(self, cpu_request: str, memory_request: str):
        """Set resource requests"""
        self.resources = {
            "requests": {"cpu": cpu_request, "memory": memory_request}
        }
    
    def generate_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes manifest"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": self.app_name},
            "spec": {
                "replicas": self.replicas,
                "selector": {"matchLabels": {"app": self.app_name}},
                "template": {
                    "metadata": {"labels": {"app": self.app_name}},
                    "spec": {
                        "containers": [{
                            "name": self.app_name,
                            "image": self.image,
                            "resources": self.resources
                        }]
                    }
                }
            }
        }


class HealthCheckConfiguration:
    """Health check configuration"""
    
    def __init__(self):
        self.startup_probe = {}
        self.liveness_probe = {}
        self.readiness_probe = {}
    
    def set_startup_probe(self, initial_delay: int, timeout: int, period: int):
        """Set startup probe"""
        self.startup_probe = {
            "initialDelaySeconds": initial_delay,
            "timeoutSeconds": timeout,
            "periodSeconds": period
        }
    
    def set_liveness_probe(self, initial_delay: int, timeout: int, period: int):
        """Set liveness probe"""
        self.liveness_probe = {
            "initialDelaySeconds": initial_delay,
            "timeoutSeconds": timeout,
            "periodSeconds": period
        }
    
    def set_readiness_probe(self, initial_delay: int, timeout: int, period: int):
        """Set readiness probe"""
        self.readiness_probe = {
            "initialDelaySeconds": initial_delay,
            "timeoutSeconds": timeout,
            "periodSeconds": period
        }


class HelmChart:
    """Helm chart configuration"""
    
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.values = {}
    
    def set_value(self, key: str, value: Any):
        """Set chart value"""
        self.values[key] = value
    
    def generate_chart_yaml(self) -> str:
        """Generate Chart.yaml"""
        return f"""apiVersion: v2
name: {self.name}
version: {self.version}
type: application
"""
    
    def generate_values_yaml(self) -> str:
        """Generate values.yaml"""
        return json.dumps(self.values, indent=2)


class DeploymentPipeline:
    """Deployment pipeline configuration"""
    
    def __init__(self):
        self.stages = []
        self.environments = {}
    
    def add_stage(self, name: str, commands: List[str]):
        """Add pipeline stage"""
        self.stages.append({"name": name, "commands": commands})
    
    def add_environment(self, name: str, config: Dict[str, Any]):
        """Add environment configuration"""
        self.environments[name] = config
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration"""
        return {
            "stages": self.stages,
            "environments": self.environments
        }


class RegistryConfiguration:
    """Container registry configuration"""
    
    def __init__(self, registry_url: str):
        self.registry_url = registry_url
        self.credentials = {}
        self.images = []
    
    def set_credentials(self, username: str, password: str):
        """Set registry credentials"""
        self.credentials = {"username": username, "password": password}
    
    def add_image(self, image_name: str, tag: str):
        """Add image to registry"""
        self.images.append(f"{self.registry_url}/{image_name}:{tag}")
    
    def get_registry_config(self) -> Dict[str, Any]:
        """Get registry configuration"""
        return {
            "registry_url": self.registry_url,
            "credentials": {"username": "***", "password": "***"},
            "images": self.images
        }


class DeploymentManager:
    """Deployment manager"""
    
    def __init__(self):
        self.docker_config = None
        self.kubernetes_deployment = None
        self.helm_chart = None
        self.registry_config = None
    
    def setup_docker(self, config: DockerConfig):
        """Setup Docker configuration"""
        self.docker_config = config
    
    def setup_kubernetes(self, deployment: KubernetesDeployment):
        """Setup Kubernetes deployment"""
        self.kubernetes_deployment = deployment
    
    def setup_helm(self, chart: HelmChart):
        """Setup Helm chart"""
        self.helm_chart = chart
    
    def setup_registry(self, config: RegistryConfiguration):
        """Setup registry"""
        self.registry_config = config
    
    def get_deployment_summary(self) -> Dict[str, Any]:
        """Get deployment summary"""
        return {
            "docker_configured": self.docker_config is not None,
            "kubernetes_configured": self.kubernetes_deployment is not None,
            "helm_configured": self.helm_chart is not None,
            "registry_configured": self.registry_config is not None
        }
