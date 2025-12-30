module.exports = {
  apps: [
    {
      name: "synth_live",
      interpreter: "python3",
      script: "./neurons/miner.py",
      args: "--netuid 50 --logging.debug --logging.trace --wallet.name bt --wallet.hotkey live --axon.port 8091 --blacklist.force_validator_permit true --blacklist.validator_min_stake 1000",
      env: {
        PYTHONPATH: ".",
      },
    },
  ],
};
