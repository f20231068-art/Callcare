from carecall.telephony.tunnel import set_env_value


def test_set_env_value_replaces_the_line_and_leaves_the_rest(tmp_path):
    f = tmp_path / ".env"
    f.write_text("A=1\nPUBLIC_BASE_URL=\nB=2\n")
    set_env_value(f, "PUBLIC_BASE_URL", "https://x.ngrok-free.app")
    assert f.read_text() == "A=1\nPUBLIC_BASE_URL=https://x.ngrok-free.app\nB=2\n"


def test_set_env_value_appends_when_missing(tmp_path):
    f = tmp_path / ".env"
    f.write_text("A=1")
    set_env_value(f, "PUBLIC_BASE_URL", "https://y")
    assert f.read_text() == "A=1\nPUBLIC_BASE_URL=https://y\n"
