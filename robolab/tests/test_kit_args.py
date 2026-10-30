from robolab.kit_args import DISABLE_EXTENSION_FS_WATCHER, disable_extension_fs_watcher


def test_disables_watcher_when_no_kit_args():
    assert disable_extension_fs_watcher("") == DISABLE_EXTENSION_FS_WATCHER
    assert disable_extension_fs_watcher(None) == DISABLE_EXTENSION_FS_WATCHER


def test_preserves_existing_tokens_and_is_idempotent():
    once = disable_extension_fs_watcher("--/renderer/activeGpu=1 --/renderer/multiGpu/enabled=false")
    assert once == "--/renderer/activeGpu=1 --/renderer/multiGpu/enabled=false " + DISABLE_EXTENSION_FS_WATCHER
    assert disable_extension_fs_watcher(once) == once


def test_does_not_override_explicit_watcher_choice():
    explicit = "--/app/extensions/fsWatcherEnabled=true --/renderer/activeGpu=1"
    assert disable_extension_fs_watcher(explicit) == explicit
