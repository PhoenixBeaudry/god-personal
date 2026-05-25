# Trainer Operations

Host-level trainer deployment helpers live here.

- `trainer.service` is a systemd unit for running the trainer API on a trainer host.
- `install_trainer_service.sh` installs and starts that systemd unit from `/root/G.O.D`.

These files are operational conveniences, not trainer runtime code.
