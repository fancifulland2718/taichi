"""Cold reconstruction sources for explicitly detachable native recordings.

These are process-local provider descriptions, not executables or serialized
Python factories. Only freeze and materialization inspect them.
"""


class FrozenNativeRecipeSource:
    def append_to_graph(self, builder, *, admission):
        """Lower the frozen region at materialization, never during replay.

        Providers may append several physical dispatches for one semantic
        source. Existing single-executable sources retain their old behavior.
        """
        builder._append_native_executable(self.materialize(), admission=admission)

    def materialize(self):
        """Return an executable owning its independent native resource leases."""
        raise NotImplementedError
