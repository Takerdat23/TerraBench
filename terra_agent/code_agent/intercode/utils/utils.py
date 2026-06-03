import docker, re, signal, time

from docker.client import DockerClient
from docker.models.containers import Container

TIMEOUT_DURATION = 10
START_UP_DELAY = 3
RELEVANT_CONTAINER_KWARGS = {
    "command",
    "environment",
    "mem_limit",
    "nano_cpus",
    "pids_limit",
    "ports",
    "volumes",
}


class timeout:
    def __init__(self, seconds=TIMEOUT_DURATION, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def get_container(ctr_name: str, image_name: str, **kwargs) -> Container:
    """
    Reset docker container with given name, or create new container with given name if it does not exist

    Returns:
        Container: reference to docker container object
    """
    client = docker.from_env()
    image = client.images.get(image_name)
    filtered_kwargs = _filter_container_kwargs(kwargs)
    all_containers = [container.name for container in client.containers.list(all=True)]
    container = None
    if ctr_name in all_containers:
        # Reuse the named container unless its image or runtime config is outdated.
        container = client.containers.get(ctr_name)
        container.reload()
        if container.image.id != image.id or _container_config_changed(container, filtered_kwargs):
            container.remove(force=True)
            container = None
        elif container.status != "running":
            container.start()
    if container is None:
        # Create + return new container from custom image
        container = client.containers.run(
            image=image,
            name=ctr_name,
            detach=True,
            tty=True,
            **filtered_kwargs)
    time.sleep(START_UP_DELAY)
    
    # Check if container was created successfully
    if not container:
        raise RuntimeError(f"Failed to create and start `{ctr_name}` container successfully")
    
    return container


def _filter_container_kwargs(kwargs):
    return {key: value for key, value in kwargs.items() if key in RELEVANT_CONTAINER_KWARGS}


def _container_config_changed(container: Container, desired_kwargs) -> bool:
    host_config = ((getattr(container, "attrs", None) or {}).get("HostConfig", {}) or {})
    desired_ports = desired_kwargs.get("ports") or {}
    if "ports" in desired_kwargs:
        current_ports = host_config.get("PortBindings") or {}
        if _normalize_port_bindings(current_ports) != _normalize_port_bindings(desired_ports):
            return True
    if _normalize_nano_cpus(host_config.get("NanoCpus")) != _normalize_nano_cpus(desired_kwargs.get("nano_cpus")):
        return True
    if _normalize_memory_limit(host_config.get("Memory")) != _normalize_memory_limit(desired_kwargs.get("mem_limit")):
        return True
    if _normalize_pids_limit(host_config.get("PidsLimit")) != _normalize_pids_limit(desired_kwargs.get("pids_limit")):
        return True
    if "volumes" in desired_kwargs:
        current_binds = host_config.get("Binds") or []
        if _normalize_volume_bindings(current_binds) != _normalize_volume_bindings(desired_kwargs.get("volumes")):
            return True
    if "environment" in desired_kwargs:
        config = ((getattr(container, "attrs", None) or {}).get("Config", {}) or {})
        if not _environment_contains(config.get("Env") or [], desired_kwargs.get("environment")):
            return True
    return False


def _normalize_port_bindings(bindings):
    normalized = {}
    for container_port, host_value in bindings.items():
        key = str(container_port)
        if isinstance(host_value, list):
            if not host_value:
                normalized[key] = None
                continue
            entry = host_value[0] or {}
            normalized[key] = str(entry.get("HostPort")) if entry.get("HostPort") is not None else None
        else:
            normalized[key] = str(host_value) if host_value is not None else None
    return normalized


def _normalize_nano_cpus(value):
    if value in (None, "", 0, "0"):
        return 0
    return int(value)


def _normalize_pids_limit(value):
    if value in (None, "", 0, "0", -1, "-1"):
        return 0
    return int(value)


def _normalize_memory_limit(value):
    if value in (None, "", 0, "0", "0b"):
        return 0
    if isinstance(value, (int, float)):
        return int(value)

    text = str(value).strip().lower()
    if not text or text in {"0", "0b"}:
        return 0
    if text.isdigit():
        return int(text)

    match = re.fullmatch(r"(\d+)([a-z]+)", text)
    if not match:
        return text

    amount = int(match.group(1))
    unit = match.group(2)
    multiplier = {
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "ki": 1024,
        "kib": 1024,
        "m": 1024 ** 2,
        "mb": 1024 ** 2,
        "mi": 1024 ** 2,
        "mib": 1024 ** 2,
        "g": 1024 ** 3,
        "gb": 1024 ** 3,
        "gi": 1024 ** 3,
        "gib": 1024 ** 3,
        "t": 1024 ** 4,
        "tb": 1024 ** 4,
        "ti": 1024 ** 4,
        "tib": 1024 ** 4,
    }.get(unit)
    if multiplier is None:
        return text
    return amount * multiplier


def _normalize_volume_bindings(bindings):
    if not bindings:
        return ()
    normalized = []
    if isinstance(bindings, dict):
        for host_path, spec in bindings.items():
            if isinstance(spec, dict):
                bind_path = spec.get("bind")
                mode = spec.get("mode") or "rw"
            else:
                bind_path = str(spec)
                mode = "rw"
            normalized.append((str(host_path), str(bind_path), str(mode)))
    else:
        for binding in bindings:
            host_path, bind_path, mode = _parse_bind_string(str(binding))
            normalized.append((host_path, bind_path, mode))
    return tuple(sorted(normalized))


def _parse_bind_string(binding):
    parts = binding.split(":")
    if len(parts) >= 3:
        host_path = ":".join(parts[:-2])
        bind_path = parts[-2]
        mode = parts[-1]
    elif len(parts) == 2:
        host_path, bind_path = parts
        mode = "rw"
    else:
        host_path = binding
        bind_path = ""
        mode = "rw"
    return host_path, bind_path, mode


def _environment_contains(current_env, desired_env):
    desired = _normalize_environment(desired_env)
    if not desired:
        return True
    current = _normalize_environment(current_env)
    return all(current.get(key) == value for key, value in desired.items())


def _normalize_environment(environment):
    if not environment:
        return {}
    if isinstance(environment, dict):
        return {str(key): str(value) for key, value in environment.items()}
    normalized = {}
    for entry in environment:
        key, _, value = str(entry).partition("=")
        if key:
            normalized[key] = value
    return normalized
