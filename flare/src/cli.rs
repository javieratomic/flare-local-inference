use std::{
    net::{IpAddr, SocketAddr},
    num::NonZeroUsize,
    path::PathBuf,
};

use clap::{Args, Parser, Subcommand, builder::PossibleValuesParser, builder::TypedValueParser};
use flare_helios::{
    engram::Storage,
    offload::{PrefixCacheOffload, Transport, parse_size},
    runtime_package::{KvCacheLimit, LoadOptions, MemoryLimit},
};

#[derive(Debug, Parser)]
#[command(
    name = "flare",
    version,
    about = "Serve prepared models on NVIDIA GPUs"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Serve a local package or download a prepared package from Hugging Face.
    Run(Box<Run>),
    /// Download a prepared package from the hardware's Hugging Face revision.
    Download(Download),
    /// Summarize a package's metadata, with optional memory, graph, and frontend details.
    Inspect(Inspect),
    /// Run this host's agent, which pairs with controllers and supervises serving ranks.
    Agent(AgentArgs),
    /// Control an orbit of agents from this machine.
    Orbit(OrbitArgs),
    #[cfg(unix)]
    #[command(hide = true)]
    CheckpointConnect {
        #[arg(long)]
        grant: PathBuf,
        #[arg(long)]
        rank: usize,
        #[arg(long)]
        stream: usize,
        #[arg(long)]
        fd: i32,
    },
    /// Manage remembered publisher trust.
    Trust {
        #[command(subcommand)]
        command: TrustCommand,
    },
}

#[derive(Debug, Args, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct HubOptions {
    /// Hub revision; defaults to gb10 or gb202 detected from the selected GPUs.
    #[arg(long, value_name = "REVISION", help_heading = "Download options")]
    pub revision: Option<String>,

    /// Package filename; required when the revision contains several .safetensors files.
    #[arg(long, value_name = "FILE", help_heading = "Download options")]
    pub filename: Option<String>,

    /// Hugging Face hub cache [default: ~/.cache/huggingface/hub].
    #[arg(long, value_name = "PATH", help_heading = "Download options")]
    pub cache_dir: Option<PathBuf>,

    /// Hugging Face access token file; omit for public repositories.
    #[arg(long, value_name = "PATH", help_heading = "Download options")]
    pub token_file: Option<PathBuf>,

    /// Use only cached files, without contacting Hugging Face.
    #[arg(long, help_heading = "Download options")]
    pub offline: bool,
}

#[derive(Debug, Args)]
pub struct Inspect {
    /// Local prepared .safetensors package.
    #[arg(value_name = "PACKAGE")]
    pub package: PathBuf,

    /// Print the selected sections as JSON.
    #[arg(long)]
    pub json: bool,

    /// Disable terminal styling.
    #[arg(long, conflicts_with = "json")]
    pub plain: bool,

    /// Include allocation addresses, reservations, and execution buffer views.
    #[arg(long)]
    pub memory_map: bool,

    /// Include graph capacities, timings, and node counts for every selected rank.
    #[arg(long)]
    pub graphs: bool,

    /// Include tokenizer, parser, vision, asset, and auxiliary table details.
    #[arg(long)]
    pub frontend: bool,

    /// Include every detail section and the complete serving contract.
    #[arg(long)]
    pub all: bool,

    /// Read only primary-file metadata without checking signatures or opening companions.
    #[arg(long, conflicts_with_all = ["publisher_keys", "trust_store"])]
    pub unverified: bool,

    /// Verify only this rank of a multi-host package whose copy holds that rank's bytes alone.
    #[arg(long, value_name = "N")]
    pub rank: Option<usize>,

    /// Additional publisher verification profiles (JSON).
    /// Development packages default to ~/.config/flare/development-publisher.json.
    #[arg(long, value_name = "PATH", action = clap::ArgAction::Append)]
    pub publisher_keys: Vec<PathBuf>,

    /// Persistent trust file whose remembered keys also verify [default: ~/.config/flare/trust.toml].
    #[arg(long, value_name = "PATH")]
    pub trust_store: Option<PathBuf>,
}

#[derive(Debug, Args)]
#[command(args_override_self = true)]
pub struct Download {
    /// Hugging Face repository containing prepared Flare packages, optionally ORG/NAME@REVISION.
    #[arg(value_name = "REPOSITORY")]
    pub repository: String,

    /// GPUs used to detect the hardware revision; unnecessary with --revision.
    #[arg(long, value_name = "N[,N...]", value_delimiter = ',', action = clap::ArgAction::Set,
        default_value = "0", value_parser = clap::value_parser!(i32).range(0..))]
    pub gpus: Vec<i32>,

    #[command(flatten)]
    pub hub: HubOptions,
}

#[derive(Debug, Args)]
pub struct AgentArgs {
    #[command(subcommand)]
    pub command: Option<AgentCommand>,

    /// Address and port to listen on.
    #[arg(long, value_name = "IP:PORT", default_value = "0.0.0.0:29900")]
    pub listen: String,
}

#[derive(Debug, Subcommand)]
pub enum AgentCommand {
    /// Install and restart this host's agent service, and prepare controller pairing.
    Install {
        /// One-time code a controller presents to pair.
        #[arg(long, value_name = "CODE")]
        join_code: Option<String>,
        /// Pair the first controller that connects, without a code.
        #[arg(long)]
        accept_first: bool,
        /// Address and port the installed agent listens on.
        #[arg(long, value_name = "IP:PORT", default_value = "0.0.0.0:29900")]
        listen: String,
    },
    /// Remove this host's authorized controllers and pairing codes; keep its key and cache.
    Reset,
}

#[derive(Debug, Args)]
pub struct OrbitArgs {
    #[command(subcommand)]
    pub command: OrbitCommand,

    /// The orbit file listing clusters and nodes [default: ~/.config/flare/orbit.toml].
    #[arg(long, value_name = "PATH", global = true)]
    pub orbit_file: Option<PathBuf>,

    /// This controller's private key [default: ~/.config/flare/orbit/controller.key].
    #[arg(long, value_name = "PATH", global = true)]
    pub key_file: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
pub enum OrbitCommand {
    /// Internal Ignition request: sign a fabric route for an admitted checkpoint source.
    #[command(hide = true)]
    CheckpointRoute {
        #[arg(long)]
        cluster: Option<String>,
    },
    /// Pair this controller with the agents on HOSTS.
    Pair {
        #[arg(required = true, value_name = "HOST")]
        hosts: Vec<String>,
        /// The join code shown when the agent was installed.
        #[arg(long, value_name = "CODE")]
        code: Option<String>,
    },
    /// Manage the controllers a paired agent trusts.
    Controllers {
        #[command(subcommand)]
        command: ControllersCommand,
    },
    /// Show each node's capacity, groups, and package cache.
    Status {
        #[arg(long, value_name = "NAME")]
        cluster: Option<String>,
    },
    /// Launch a package across nodes or start a saved group.
    Run(Box<OrbitRun>),
    /// Show the live metrics dashboard for every group and GPU on the cluster's nodes.
    Dashboard {
        #[arg(long, value_name = "NAME")]
        cluster: Option<String>,
    },
    /// List saved groups and retained runs on the cluster's nodes.
    Ls {
        #[arg(long, value_name = "NAME")]
        cluster: Option<String>,
    },
    /// Follow retained and live rank output; Ctrl-C detaches without stopping ranks.
    Attach {
        #[arg(long, value_name = "NAME")]
        cluster: Option<String>,
        /// Only the group with this name or identifier.
        #[arg(value_name = "GROUP")]
        group: Option<String>,
    },
    /// Stop running groups.
    Stop {
        #[arg(long, value_name = "NAME")]
        cluster: Option<String>,
        /// Only the group with this name or identifier.
        #[arg(value_name = "GROUP")]
        group: Option<String>,
    },
    /// Remove a stopped group's saved definition and retained run history.
    Rm {
        /// Group name or exact launch identifier; a launch identifier keeps its saved definition.
        group: String,
    },
    /// Remove a node from the orbit file.
    Forget {
        #[arg(value_name = "NODE")]
        node: String,
    },
    /// Stage package data on nodes without starting ranks.
    Push(Box<OrbitPush>),
    /// List what every node's package cache holds.
    Packages {
        #[arg(long, value_name = "NAME")]
        cluster: Option<String>,
    },
}

#[derive(Debug, Args)]
pub struct OrbitPush {
    /// A local package path or NODE:/absolute/path on a node.
    #[arg(value_name = "PACKAGE")]
    pub package: String,

    /// Target nodes: one per rank in rank order, or any number to receive the complete package.
    #[arg(
        long,
        value_name = "NODE[,NODE...]",
        value_delimiter = ',',
        required = true
    )]
    pub to: Vec<String>,

    /// Stream package data through fabric interfaces in the target nodes' ring order.
    #[arg(long)]
    pub roce_ring: bool,

    /// Parallel TCP streams per destination file over the ring fabric [default: 8].
    #[arg(long, value_name = "N", requires = "roce_ring", value_parser = clap::value_parser!(u8).range(1..=64))]
    pub transfer_streams: Option<u8>,

    #[command(flatten)]
    pub serving: ServingOptions,
}

impl OrbitPush {
    /// What package verification and publisher acceptance need, as `flare run` takes them.
    pub fn run_args(&self) -> Run {
        Run {
            package: PathBuf::from(&self.package),
            host: "127.0.0.1".parse().unwrap(),
            port: 8000,
            served_model_name: None,
            roce_ring: self.roce_ring,
            rendezvous: None,
            rank: None,
            api_key_file: None,
            scheduler_policy: "balanced".into(),
            serving: self.serving.clone(),
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum ControllersCommand {
    /// Authorize another controller on every paired node.
    Add {
        name: String,
        /// The controller's public key, in hex.
        key: String,
    },
    /// Revoke a controller on every paired node.
    Remove {
        /// The controller's public key, in hex.
        key: String,
    },
}

#[derive(Debug, Args)]
#[command(args_override_self = true)]
pub struct OrbitRun {
    /// A Hugging Face repository, local package path, or NODE:/absolute/path; omitted when --name names a
    /// [[group]] of the orbit file, which supplies it.
    #[arg(value_name = "PACKAGE")]
    pub package: Option<String>,

    /// Cluster whose nodes serve the ranks in order; without it placement chooses from every
    /// node of the orbit.
    #[arg(long, value_name = "NAME")]
    pub cluster: Option<String>,

    /// The group's name, used by attach, stop and ls [default: the cluster's name, or the
    /// package's file name].
    #[arg(long, value_name = "NAME")]
    pub name: Option<String>,

    /// Save this named group's launch configuration for reuse after stopping or rebooting.
    #[arg(long, requires = "name")]
    pub persistent: bool,

    /// Candidate nodes in rank order, instead of a cluster or the whole orbit.
    #[arg(long, value_name = "NODE[,NODE...]", value_delimiter = ',')]
    pub nodes: Vec<String>,

    /// Verify package metadata and print placement without starting ranks or saving a group.
    #[arg(long)]
    pub dry_run: bool,

    /// Return once rank processes start; the HTTP server may still be loading the model.
    #[arg(long)]
    pub detach: bool,

    /// Upload this Flare executable [default: saved override or each node's installed Flare].
    #[arg(long, value_name = "PATH")]
    pub executable: Option<PathBuf>,

    /// Experimental ring in rank order: only neighbor RoCE QPs connect;
    /// native CPU relays forward non-neighbor traffic. Also emulates a ring on a switch.
    #[arg(long, help_heading = "Multi-host options")]
    pub roce_ring: bool,

    /// Parallel TCP streams per destination file over the ring fabric [default: saved group or 8].
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u8).range(1..=64), help_heading = "Multi-host options")]
    pub transfer_streams: Option<u8>,

    /// IP address for rank zero's HTTP server [default: saved group or 127.0.0.1].
    #[arg(long, value_name = "ADDR", help_heading = "Serving options")]
    pub host: Option<IpAddr>,

    /// Port for rank zero's HTTP server [default: saved group or 8000].
    #[arg(long, value_name = "N", help_heading = "Serving options")]
    pub port: Option<u16>,

    /// Replace the package's serving alias; the canonical model ID remains accepted.
    #[arg(long, value_name = "ALIAS", value_parser = served_model_name, help_heading = "Serving options")]
    pub served_model_name: Option<String>,

    /// Favor throughput or output cadence [default: saved group or balanced].
    #[arg(long, value_name = "MODE",
        value_parser = ["balanced", "throughput"], help_heading = "Serving options")]
    pub scheduler_policy: Option<String>,

    /// File on rank zero's node containing the serving API key; omit to disable authentication.
    #[arg(long, value_name = "PATH", help_heading = "Serving options")]
    pub api_key_file: Option<PathBuf>,

    #[command(flatten)]
    pub serving: ServingOptions,
}

impl OrbitRun {
    /// Serving options every rank loads with.
    pub fn rank_options(&self, matches: &clap::ArgMatches) -> Vec<String> {
        let mut options = self.serving.rank_options(self.roce_ring);
        // Explicit defaults must override saved values, while omitted defaults must not.
        if self.serving.kv_cache_limit == KvCacheLimit::Memory
            && matches.value_source("kv_cache_limit")
                == Some(clap::parser::ValueSource::CommandLine)
        {
            options.extend(["--kv-cache-limit".into(), "memory".into()]);
        }
        if self.serving.embedding_table_cache_size == 0
            && matches.value_source("embedding_table_cache_size")
                == Some(clap::parser::ValueSource::CommandLine)
        {
            options.extend(["--embedding-table-cache-size".into(), "0".into()]);
        }
        options
    }

    /// Options only rank zero's HTTP server takes.
    pub fn rank_zero_options(&self) -> Vec<String> {
        let mut options = Vec::new();
        if let Some(host) = self.host {
            options.extend(["--host".into(), host.to_string()]);
        }
        if let Some(port) = self.port {
            options.extend(["--port".into(), port.to_string()]);
        }
        if let Some(name) = &self.served_model_name {
            options.extend(["--served-model-name".into(), name.clone()]);
        }
        if let Some(policy) = &self.scheduler_policy {
            options.extend(["--scheduler-policy".into(), policy.clone()]);
        }
        if let Some(path) = &self.api_key_file {
            options.extend(["--api-key-file".into(), path.display().to_string()]);
        }
        options
    }

    /// The equivalent single-host arguments, for publisher trust resolution of the package.
    pub fn run_args(&self) -> Run {
        Run {
            package: PathBuf::from(self.package.as_deref().unwrap_or_default()),
            host: self.host.unwrap_or_else(|| "127.0.0.1".parse().unwrap()),
            port: self.port.unwrap_or(8000),
            served_model_name: self.served_model_name.clone(),
            roce_ring: self.roce_ring,
            rendezvous: None,
            rank: None,
            api_key_file: self.api_key_file.clone(),
            scheduler_policy: self
                .scheduler_policy
                .clone()
                .unwrap_or_else(|| "balanced".into()),
            serving: self.serving.clone(),
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum TrustCommand {
    /// Remove all remembered publisher identities for a signing key.
    Revoke {
        fingerprint: String,
        #[arg(long, value_name = "PATH")]
        trust_store: Option<PathBuf>,
    },
}

#[derive(Debug, Args)]
#[command(
    args_override_self = true,
    after_help = "Examples:\n  flare run model.safetensors --gpus 0 --host 127.0.0.1 --port 8000\n  flare run model.safetensors --embedding-table-storage ram --mtp-draft-tokens 3\n\nConcurrency and draft limits must fit the package's prepared capacities.\nOmitted concurrency and speculation options use package defaults."
)]
pub struct Run {
    /// Local prepared .safetensors package or Hugging Face ORG/NAME[@REVISION].
    #[arg(value_name = "PACKAGE")]
    pub package: PathBuf,

    /// IP address for the HTTP server.
    #[arg(
        long,
        value_name = "ADDR",
        default_value = "127.0.0.1",
        help_heading = "Serving options"
    )]
    pub host: IpAddr,

    /// Port for the HTTP server.
    #[arg(
        long,
        value_name = "N",
        default_value = "8000",
        help_heading = "Serving options"
    )]
    pub port: u16,

    /// Replace the package's serving alias; the canonical model ID remains accepted.
    #[arg(long, value_name = "ALIAS", value_parser = served_model_name, help_heading = "Serving options")]
    pub served_model_name: Option<String>,

    /// Experimental ring in rank order: only neighbor RoCE QPs connect;
    /// native CPU relays forward non-neighbor traffic. Also emulates a ring on a switch.
    #[arg(long, help_heading = "Multi-host options")]
    pub roce_ring: bool,

    /// Rank zero's rendezvous address: where rank zero listens, or on a peer, where it joins rank zero.
    #[arg(long, value_name = "HOST:PORT", help_heading = "Multi-host options")]
    pub rendezvous: Option<String>,

    /// This host's rank of a multi-host package.
    #[arg(long, value_name = "N", help_heading = "Multi-host options")]
    pub rank: Option<usize>,

    /// File containing the serving API key; omit to disable authentication.
    #[arg(long, value_name = "PATH", help_heading = "Serving options")]
    pub api_key_file: Option<PathBuf>,

    /// Favor token throughput or balance it with output cadence; both retain token-service fairness.
    #[arg(long, value_name = "MODE", default_value = "balanced",
        value_parser = ["balanced", "throughput"], help_heading = "Serving options")]
    pub scheduler_policy: String,

    #[command(flatten)]
    pub serving: ServingOptions,
}

/// Options that select a package's trust, loading, speculation and caching; every rank of a
/// multi-host package loads with the same values.
#[derive(Debug, Args, Clone)]
pub struct ServingOptions {
    /// Additional publisher verification profiles (JSON); does not grant execution trust.
    /// Development packages default to ~/.config/flare/development-publisher.json.
    #[arg(long, value_name = "PATH", action = clap::ArgAction::Append)]
    pub publisher_keys: Vec<PathBuf>,
    /// Verify stored allocation and table ranges before loading model GPU resources.
    #[arg(long)]
    pub verify_weights: bool,
    /// Permanently accept this exact signing key together with --dangerously-accept-publisher.
    #[arg(
        long,
        requires = "dangerously_accept_publisher",
        value_name = "FINGERPRINT"
    )]
    pub dangerously_approve_key: Option<String>,
    /// Exact Hugging Face publisher namespace acknowledged for unattended first use.
    #[arg(long, requires = "dangerously_approve_key", value_name = "NAMESPACE")]
    pub dangerously_accept_publisher: Option<String>,
    /// Persistent trust file [default: ~/.config/flare/trust.toml].
    #[arg(long, value_name = "PATH")]
    pub trust_store: Option<PathBuf>,

    #[command(flatten)]
    pub hub: HubOptions,

    /// GPUs in rank order, relative to CUDA_VISIBLE_DEVICES when set.
    #[arg(long, value_name = "N[,N...]", value_delimiter = ',', action = clap::ArgAction::Set,
        default_value = "0", value_parser = clap::value_parser!(i32).range(0..),
        help_heading = "Serving options")]
    pub gpus: Vec<i32>,

    /// Limit KV capacity by available memory, one context, or context times concurrency.
    #[arg(
        long,
        default_value = "memory",
        value_name = "MODE",
        help_heading = "Serving options",
        value_parser = PossibleValuesParser::new(["concurrency", "context", "memory"])
            .map(|value| match value.as_str() {
                "concurrency" => KvCacheLimit::Concurrency,
                "context" => KvCacheLimit::Context,
                _ => KvCacheLimit::Memory,
            })
    )]
    pub kv_cache_limit: KvCacheLimit,

    /// Per-GPU serving memory: occupied size (30G), free reserve (-30G), or total fraction (0.8).
    /// Omitted: package serving reserve.
    #[arg(
        long,
        allow_hyphen_values = true,
        value_name = "LIMIT",
        help_heading = "Serving options"
    )]
    pub memory_limit: Option<MemoryLimit>,

    /// Maximum simultaneous requests [default: package setting].
    #[arg(long, value_name = "N", help_heading = "Serving options")]
    pub max_concurrency: Option<NonZeroUsize>,

    /// Storage for auxiliary embedding tables (PLE/Engram) [default: disk].
    #[arg(long, value_name = "MODE", help_heading = "Serving options",
        value_parser = PossibleValuesParser::new(["disk", "ram"])
            .map(|value| if value == "disk" { Storage::Disk } else { Storage::Ram }))]
    pub embedding_table_storage: Option<Storage>,

    /// Per-host budget for a disk-backed embedding row cache, e.g. 1G; includes the
    /// index, usage counters and miss workspace. Selects disk storage; 0 disables [default: 0].
    #[arg(long, value_name = "SIZE", value_parser = parse_size, default_value = "0", help_heading = "Serving options")]
    pub embedding_table_cache_size: u64,

    /// Retain embedding scales in host RAM while weights remain on disk; adds to the row-cache budget.
    #[arg(long, help_heading = "Serving options")]
    pub embedding_table_resident_scales: bool,

    /// Disk workers per embedding table; shares a fixed I/O budget [default: 4 on Windows, 1 elsewhere].
    #[arg(long, value_name = "N", help_heading = "Advanced serving options",
        value_parser = PossibleValuesParser::new(["1", "2", "4", "8"]).map(|value| value.parse::<usize>().unwrap()))]
    pub embedding_table_workers: Option<usize>,

    /// Maximum drafts per round; 0 disables speculation [default: package setting].
    #[arg(long, value_name = "N", help_heading = "Speculation options")]
    pub mtp_draft_tokens: Option<usize>,

    /// Advanced: fix the total verification-row budget, including pending inputs, using
    /// one exact captured packed graph per batch [default: adaptive within captured capacities].
    /// Requires adaptive prefix selection at the captured draft depth.
    #[arg(long, value_name = "N", help_heading = "Advanced speculation options")]
    pub mtp_verification_token_budget: Option<NonZeroUsize>,

    /// Advanced: retain all captured packed verification capacities up to N total rows,
    /// including pending inputs [default: all captured capacities]. Requires adaptive
    /// prefix selection at the captured draft depth.
    #[arg(
        long,
        value_name = "N",
        conflicts_with = "mtp_verification_token_budget",
        help_heading = "Advanced speculation options"
    )]
    pub mtp_max_verification_token_budget: Option<NonZeroUsize>,

    /// Adapt verification width to acceptance estimates and execution cost [default: package setting].
    #[arg(
        long,
        overrides_with = "no_mtp_adaptive_verification",
        help_heading = "Speculation options"
    )]
    pub mtp_adaptive_verification: bool,

    /// Use the maximum width where captured, otherwise width one [default: package setting].
    #[arg(
        long,
        overrides_with = "mtp_adaptive_verification",
        help_heading = "Speculation options"
    )]
    pub no_mtp_adaptive_verification: bool,

    /// Host memory for prefix checkpoints and prepared inputs, e.g. 16G; 0 disables [default: disabled].
    #[arg(long, value_name = "SIZE", value_parser = parse_size, help_heading = "Prefix cache options")]
    pub prefix_cache_memory: Option<u64>,

    /// Directory for prefix checkpoints and prepared inputs on disk [default: disabled].
    #[arg(long, value_name = "DIR", help_heading = "Prefix cache options")]
    pub prefix_cache_disk: Option<PathBuf>,

    /// Disk retention limit, e.g. 100G; 0 disables [default: 10% of free space].
    #[arg(long, value_name = "SIZE", value_parser = parse_size, help_heading = "Prefix cache options")]
    pub prefix_cache_disk_limit: Option<u64>,

    /// Transfer disk checkpoints through host staging or GPUDirect Storage [default: staged].
    #[arg(long, value_name = "MODE", help_heading = "Prefix cache options",
        value_parser = PossibleValuesParser::new(["staged", "direct"])
            .map(|value| if value == "staged" { Transport::Staged } else { Transport::Direct }))]
    pub prefix_cache_disk_transport: Option<Transport>,
}

fn served_model_name(value: &str) -> Result<String, String> {
    if value.trim().is_empty() {
        Err("served model name must not be empty".into())
    } else {
        Ok(value.to_owned())
    }
}

impl std::ops::Deref for Run {
    type Target = ServingOptions;

    fn deref(&self) -> &ServingOptions {
        &self.serving
    }
}

impl Run {
    pub fn bind_address(&self) -> SocketAddr {
        SocketAddr::new(self.host, self.port)
    }

    /// Serving options every rank of a multi-host package must load with: they select the same
    /// graphs and capacities on every host.
    #[cfg(test)]
    pub fn rank_options(&self) -> Vec<String> {
        self.serving.rank_options(self.roce_ring)
    }

    pub fn load_options(&self) -> LoadOptions {
        self.serving.load_options(self.rank)
    }
}

impl ServingOptions {
    /// Serving options every rank of a multi-host package must load with.
    pub fn rank_options(&self, roce_ring: bool) -> Vec<String> {
        let mut options = Vec::new();
        if self.kv_cache_limit != KvCacheLimit::Memory {
            options.extend(["--kv-cache-limit".into(), self.kv_cache_limit.to_string()]);
        }
        if let Some(limit) = self.memory_limit {
            options.push(format!("--memory-limit={limit}"));
        }
        if roce_ring {
            options.push("--roce-ring".into());
        }
        if self.verify_weights {
            options.push("--verify-weights".into());
        }
        if let Some(n) = self.max_concurrency {
            options.extend(["--max-concurrency".into(), n.to_string()]);
        }
        if let Some(n) = self.mtp_draft_tokens {
            options.extend(["--mtp-draft-tokens".into(), n.to_string()]);
        }
        if let Some(n) = self.mtp_verification_token_budget {
            options.extend(["--mtp-verification-token-budget".into(), n.to_string()]);
        }
        if let Some(n) = self.mtp_max_verification_token_budget {
            options.extend(["--mtp-max-verification-token-budget".into(), n.to_string()]);
        }
        if self.mtp_adaptive_verification {
            options.push("--mtp-adaptive-verification".into());
        }
        if self.no_mtp_adaptive_verification {
            options.push("--no-mtp-adaptive-verification".into());
        }
        if let Some(storage) = self.embedding_table_storage {
            options.extend([
                "--embedding-table-storage".into(),
                if storage == Storage::Disk {
                    "disk"
                } else {
                    "ram"
                }
                .into(),
            ]);
        }
        if let Some(workers) = self.embedding_table_workers {
            options.extend(["--embedding-table-workers".into(), workers.to_string()]);
        }
        if self.embedding_table_cache_size != 0 {
            options.extend([
                "--embedding-table-cache-size".into(),
                self.embedding_table_cache_size.to_string(),
            ]);
        }
        if self.embedding_table_resident_scales {
            options.push("--embedding-table-resident-scales".into());
        }
        // Every rank keeps its own records under its host's directory; rank zero's store drives
        // all of them, so every rank needs the same settings.
        if let Some(bytes) = self.prefix_cache_memory {
            options.extend(["--prefix-cache-memory".into(), bytes.to_string()]);
        }
        if let Some(directory) = &self.prefix_cache_disk {
            options.extend(["--prefix-cache-disk".into(), directory.display().to_string()]);
        }
        if let Some(bytes) = self.prefix_cache_disk_limit {
            options.extend(["--prefix-cache-disk-limit".into(), bytes.to_string()]);
        }
        if let Some(transport) = self.prefix_cache_disk_transport {
            options.extend([
                "--prefix-cache-disk-transport".into(),
                match transport {
                    Transport::Staged => "staged",
                    Transport::Direct => "direct",
                }
                .into(),
            ]);
        }
        options
    }

    pub fn load_options(&self, cluster_rank: Option<usize>) -> LoadOptions {
        LoadOptions {
            kv_cache_limit: self.kv_cache_limit,
            memory_limit: self.memory_limit,
            embedding_table_storage: self.embedding_table_storage,
            embedding_table_cache_bytes: self.embedding_table_cache_size,
            embedding_table_resident_scales: self.embedding_table_resident_scales,
            embedding_table_workers: self.embedding_table_workers,
            max_concurrency: self.max_concurrency.map(NonZeroUsize::get),
            mtp_draft_tokens: self.mtp_draft_tokens,
            mtp_verification_token_budget: self
                .mtp_verification_token_budget
                .map(NonZeroUsize::get),
            mtp_max_verification_token_budget: self
                .mtp_max_verification_token_budget
                .map(NonZeroUsize::get),
            adaptive_verification: if self.mtp_adaptive_verification {
                Some(true)
            } else if self.no_mtp_adaptive_verification {
                Some(false)
            } else {
                None
            },
            device_labels: Some(self.gpus.clone()),
            prefix_cache: PrefixCacheOffload {
                memory_bytes: self.prefix_cache_memory,
                disk: self.prefix_cache_disk.clone(),
                disk_limit: self.prefix_cache_disk_limit,
                transport: self.prefix_cache_disk_transport,
            },
            cluster_rank,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_orbit_options_preserve_saved_values_and_allow_explicit_defaults() {
        use clap::{CommandFactory, FromArgMatches};
        assert!(Cli::try_parse_from(["flare", "orbit", "run", "model", "--persistent"]).is_err());
        for explicit in [false, true] {
            let mut arguments = vec!["flare", "orbit", "run", "--name", "service", "--persistent"];
            if explicit {
                arguments.extend([
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8000",
                    "--scheduler-policy",
                    "balanced",
                    "--kv-cache-limit",
                    "memory",
                    "--embedding-table-cache-size",
                    "0",
                    "--gpus",
                    "0",
                ]);
            }
            let matches = Cli::command().try_get_matches_from(arguments).unwrap();
            let Command::Orbit(orbit) = Cli::from_arg_matches(&matches).unwrap().command else {
                panic!()
            };
            let OrbitCommand::Run(orbit) = orbit.command else {
                panic!()
            };
            assert!(orbit.persistent);
            let run_matches = matches
                .subcommand_matches("orbit")
                .unwrap()
                .subcommand_matches("run")
                .unwrap();
            let mut saved = vec![
                "flare".to_string(),
                "run".into(),
                "model".into(),
                "--host".into(),
                "0.0.0.0".into(),
                "--port".into(),
                "8001".into(),
                "--scheduler-policy".into(),
                "throughput".into(),
                "--kv-cache-limit".into(),
                "context".into(),
                "--embedding-table-cache-size".into(),
                "1G".into(),
            ];
            saved.extend(orbit.rank_options(run_matches));
            saved.extend(orbit.rank_zero_options());
            let Command::Run(effective) = Cli::try_parse_from(saved).unwrap().command else {
                panic!()
            };
            assert_eq!(effective.port, if explicit { 8000 } else { 8001 });
            assert_eq!(
                effective.host.to_string(),
                if explicit { "127.0.0.1" } else { "0.0.0.0" }
            );
            assert_eq!(
                effective.scheduler_policy,
                if explicit { "balanced" } else { "throughput" }
            );
            assert_eq!(
                effective.serving.kv_cache_limit,
                if explicit {
                    KvCacheLimit::Memory
                } else {
                    KvCacheLimit::Context
                }
            );
            assert_eq!(
                effective.serving.embedding_table_cache_size,
                if explicit { 0 } else { 1 << 30 }
            );
            assert_eq!(
                run_matches.value_source("gpus") == Some(clap::parser::ValueSource::CommandLine),
                explicit
            );
        }
    }

    fn run(arguments: &[&str]) -> Run {
        let Command::Run(args) = Cli::try_parse_from(arguments).unwrap().command else {
            panic!("expected run")
        };
        *args
    }

    #[test]
    fn embedding_workers_reach_loader_and_remote_arguments() {
        assert_eq!(
            run(&["flare", "run", "model.safetensors"])
                .load_options()
                .embedding_table_workers,
            None
        );
        for workers in ["1", "2", "4", "8"] {
            let args = run(&[
                "flare",
                "run",
                "model.safetensors",
                "--embedding-table-workers",
                workers,
            ]);
            assert_eq!(
                args.load_options().embedding_table_workers,
                Some(workers.parse().unwrap())
            );
            assert_eq!(
                args.load_options().embedding_workers(),
                workers.parse::<usize>().unwrap()
            );
            assert!(
                args.rank_options()
                    .windows(2)
                    .any(|pair| pair == ["--embedding-table-workers", workers])
            );
        }
        for workers in ["0", "3", "5"] {
            assert!(
                Cli::try_parse_from([
                    "flare",
                    "run",
                    "model.safetensors",
                    "--embedding-table-workers",
                    workers
                ])
                .is_err()
            );
        }
    }

    #[test]
    fn omitted_limits_preserve_package_defaults() {
        let args = run(&["flare", "run", "model.safetensors"]);
        assert_eq!(args.bind_address().to_string(), "127.0.0.1:8000");
        assert_eq!(args.served_model_name, None);
        assert_eq!(args.scheduler_policy, "balanced");
        let options = args.load_options();
        assert_eq!(options.device_labels, Some(vec![0]));
        assert_eq!(options.embedding_table_storage, None);
        assert_eq!(options.embedding_table_cache_bytes, 0);
        assert_eq!(options.max_concurrency, None);
        assert_eq!(options.kv_cache_limit, KvCacheLimit::Memory);
        assert_eq!(options.memory_limit, None);
        assert_eq!(options.mtp_draft_tokens, None);
        assert_eq!(options.mtp_verification_token_budget, None);
        assert_eq!(options.mtp_max_verification_token_budget, None);
        assert_eq!(options.adaptive_verification, None);
        assert_eq!(options.prefix_cache, PrefixCacheOffload::default());
    }

    #[test]
    fn scheduling_preference_is_explicit_and_validated() {
        assert_eq!(
            run(&[
                "flare",
                "run",
                "model.safetensors",
                "--scheduler-policy",
                "throughput"
            ])
            .scheduler_policy,
            "throughput"
        );
        assert!(
            Cli::try_parse_from([
                "flare",
                "run",
                "model.safetensors",
                "--scheduler-policy",
                "unknown"
            ])
            .is_err()
        );
    }

    #[test]
    fn explicit_options_reach_the_loader() {
        let args = run(&[
            "flare",
            "run",
            "--gpus=1",
            "--gpus=2,0",
            "--host",
            "::1",
            "--port",
            "8001",
            "model.safetensors",
            "--api-key-file",
            "key.txt",
            "--max-concurrency",
            "3",
            "--mtp-draft-tokens=0",
            "--embedding-table-storage",
            "ram",
            "--prefix-cache-memory",
            "1.5GiB",
            "--prefix-cache-disk",
            "prefixes",
            "--prefix-cache-disk-limit",
            "0",
            "--prefix-cache-disk-transport",
            "direct",
        ]);
        assert_eq!(args.package, PathBuf::from("model.safetensors"));
        assert_eq!(args.bind_address().to_string(), "[::1]:8001");
        assert_eq!(args.api_key_file, Some(PathBuf::from("key.txt")));
        let options = args.load_options();
        assert_eq!(options.device_labels, Some(vec![2, 0]));
        assert_eq!(options.embedding_table_storage, Some(Storage::Ram));
        assert_eq!(options.max_concurrency, Some(3));
        assert_eq!(options.mtp_draft_tokens, Some(0));
        assert_eq!(
            options.prefix_cache,
            PrefixCacheOffload {
                memory_bytes: Some(3 << 29),
                disk: Some(PathBuf::from("prefixes")),
                disk_limit: Some(0),
                transport: Some(Transport::Direct),
            }
        );
    }

    #[test]
    fn ring_prestaging_keeps_target_order_and_selects_fabric_distribution() {
        let parsed = Cli::try_parse_from([
            "flare",
            "orbit",
            "push",
            "model.safetensors",
            "--to",
            "zero,one,two,three",
            "--roce-ring",
            "--transfer-streams",
            "8",
        ])
        .unwrap();
        let Command::Orbit(OrbitArgs {
            command: OrbitCommand::Push(push),
            ..
        }) = parsed.command
        else {
            panic!("expected prestaging command");
        };
        assert_eq!(push.to, ["zero", "one", "two", "three"]);
        assert_eq!(push.transfer_streams, Some(8));
        assert!(push.run_args().roce_ring);
    }

    #[test]
    fn transfer_stream_counts_are_bounded_and_stay_out_of_rank_arguments() {
        for count in ["1", "4", "8", "16", "32", "64"] {
            let parsed = Cli::try_parse_from([
                "flare",
                "orbit",
                "run",
                "model.safetensors",
                "--roce-ring",
                "--transfer-streams",
                count,
            ])
            .unwrap();
            let Command::Orbit(OrbitArgs {
                command: OrbitCommand::Run(run),
                ..
            }) = parsed.command
            else {
                panic!("expected orbit run");
            };
            assert_eq!(run.transfer_streams, Some(count.parse().unwrap()));
            assert!(
                !run.run_args()
                    .rank_options()
                    .iter()
                    .any(|s| s.contains("transfer-streams"))
            );
        }
        for command in ["run", "push"] {
            for count in ["0", "65", "256"] {
                let mut args = vec![
                    "flare",
                    "orbit",
                    command,
                    "m.safetensors",
                    "--roce-ring",
                    "--transfer-streams",
                    count,
                ];
                if command == "push" {
                    args.extend(["--to", "zero,one"]);
                }
                assert!(Cli::try_parse_from(args).is_err());
            }
        }
        assert!(
            Cli::try_parse_from([
                "flare",
                "orbit",
                "push",
                "m.safetensors",
                "--to",
                "zero,one",
                "--transfer-streams",
                "4"
            ])
            .is_err()
        );
    }

    #[test]
    fn ring_flag_is_forwarded_to_every_agent_started_rank() {
        let args = run(&["flare", "run", "model.safetensors", "--roce-ring"]);
        assert!(args.roce_ring);
        assert_eq!(args.rank_options(), vec!["--roce-ring"]);
    }

    #[test]
    fn verification_token_budget_reaches_loader_and_peer_ranks() {
        for budget in ["24", "32"] {
            let args = run(&[
                "flare",
                "run",
                "model.safetensors",
                "--mtp-verification-token-budget",
                budget,
            ]);
            assert_eq!(
                args.load_options().mtp_verification_token_budget,
                Some(budget.parse().unwrap())
            );
            assert_eq!(
                args.rank_options(),
                vec!["--mtp-verification-token-budget", budget]
            );
        }
        for invalid in ["0", "-1", "unbounded"] {
            assert!(
                Cli::try_parse_from([
                    "flare",
                    "run",
                    "model.safetensors",
                    "--mtp-verification-token-budget",
                    invalid
                ])
                .is_err()
            );
        }
    }

    #[test]
    fn maximum_verification_token_budget_reaches_loader_and_peer_ranks() {
        for budget in ["24", "32"] {
            let args = run(&[
                "flare",
                "run",
                "model.safetensors",
                "--mtp-max-verification-token-budget",
                budget,
            ]);
            assert_eq!(
                args.load_options().mtp_max_verification_token_budget,
                Some(budget.parse().unwrap())
            );
            assert_eq!(
                args.rank_options(),
                vec!["--mtp-max-verification-token-budget", budget]
            );
        }
        for invalid in ["0", "-1", "unbounded"] {
            assert!(
                Cli::try_parse_from([
                    "flare",
                    "run",
                    "model.safetensors",
                    "--mtp-max-verification-token-budget",
                    invalid
                ])
                .is_err()
            );
        }
        assert!(
            Cli::try_parse_from([
                "flare",
                "run",
                "model.safetensors",
                "--mtp-verification-token-budget",
                "24",
                "--mtp-max-verification-token-budget",
                "24"
            ])
            .is_err()
        );
    }

    #[test]
    fn last_verification_flag_wins() {
        let enable = "--mtp-adaptive-verification";
        let disable = "--no-mtp-adaptive-verification";
        for (flags, expected) in [
            (vec![enable], true),
            (vec![disable], false),
            (vec![enable, disable], false),
            (vec![disable, enable], true),
        ] {
            let mut arguments = vec!["flare", "run", "model.safetensors"];
            arguments.extend(flags);
            assert_eq!(
                run(&arguments).load_options().adaptive_verification,
                Some(expected)
            );
        }
    }

    #[test]
    fn listener_flags_support_independent_overrides_and_ipv6() {
        for (flags, expected) in [
            (vec!["--host", "0.0.0.0"], "0.0.0.0:8000"),
            (vec!["--port", "8001"], "127.0.0.1:8001"),
            (vec!["--host", "::1", "--port", "8002"], "[::1]:8002"),
        ] {
            let mut arguments = vec!["flare", "run", "model.safetensors"];
            arguments.extend(flags);
            assert_eq!(run(&arguments).bind_address().to_string(), expected);
        }
    }

    #[test]
    fn served_model_name_accepts_aliases_and_rejects_blank_values() {
        for name in ["my-model", "organization/model", "local model"] {
            let args = run(&[
                "flare",
                "run",
                "model.safetensors",
                "--served-model-name",
                name,
            ]);
            assert_eq!(args.served_model_name.as_deref(), Some(name));
        }
        for name in ["", " ", "\t"] {
            assert!(
                Cli::try_parse_from([
                    "flare",
                    "run",
                    "model.safetensors",
                    "--served-model-name",
                    name
                ])
                .is_err()
            );
        }
    }

    #[test]
    fn embedding_row_cache_accepts_a_binary_memory_budget() {
        let args = run(&[
            "flare",
            "run",
            "model.safetensors",
            "--embedding-table-cache-size",
            "1G",
        ]);
        let options = args.load_options();
        assert_eq!(options.embedding_table_cache_bytes, 1 << 30);
        assert_eq!(options.embedding_table_storage, None);
        assert_eq!(
            args.rank_options(),
            ["--embedding-table-cache-size", "1073741824"]
        );
    }

    #[test]
    fn prefix_cache_settings_reach_every_rank() {
        let args = run(&[
            "flare",
            "run",
            "model.safetensors",
            "--prefix-cache-memory",
            "0",
            "--prefix-cache-disk",
            "/var/cache/flare",
            "--prefix-cache-disk-limit",
            "100G",
            "--prefix-cache-disk-transport",
            "staged",
        ]);
        let options = args.rank_options();
        assert_eq!(
            options,
            [
                "--prefix-cache-memory",
                "0",
                "--prefix-cache-disk",
                "/var/cache/flare",
                "--prefix-cache-disk-limit",
                "107374182400",
                "--prefix-cache-disk-transport",
                "staged",
            ]
        );
        let mut forwarded = vec!["flare", "run", "model.safetensors"];
        forwarded.extend(options.iter().map(String::as_str));
        assert_eq!(
            run(&forwarded).load_options().prefix_cache,
            args.load_options().prefix_cache
        );
        assert!(run(&["flare", "run", "model.safetensors"]).rank_options().is_empty());
    }

    #[test]
    fn resident_embedding_scales_are_forwarded_to_every_rank() {
        let args = run(&["flare", "run", "model.safetensors", "--embedding-table-resident-scales",
                         "--embedding-table-cache-size", "1G"]);
        assert!(args.load_options().embedding_table_resident_scales);
        assert_eq!(args.load_options().embedding_table_cache_bytes, 1 << 30);
        assert!(args.rank_options().contains(&"--embedding-table-resident-scales".to_owned()));
        assert!(!run(&["flare", "run", "model.safetensors"]).load_options().embedding_table_resident_scales);
    }

    #[test]
    fn memory_limits_accept_signed_values_and_reach_every_rank() {
        for value in ["30g", "-30g", "0.8"] {
            for flags in [
                vec!["--memory-limit".to_owned(), value.to_owned()],
                vec![format!("--memory-limit={value}")],
            ] {
                let mut arguments = vec![
                    "flare",
                    "run",
                    "model.safetensors",
                    "--kv-cache-limit",
                    "context",
                ];
                arguments.extend(flags.iter().map(String::as_str));
                let args = run(&arguments);
                assert_eq!(args.load_options().kv_cache_limit, KvCacheLimit::Context);
                assert_eq!(
                    args.load_options().memory_limit,
                    Some(value.parse().unwrap())
                );
                let forwarded = args.rank_options();
                let mut peer = vec!["flare", "run", "model.safetensors"];
                peer.extend(forwarded.iter().map(String::as_str));
                let peer = run(&peer).load_options();
                assert_eq!(peer.memory_limit, args.load_options().memory_limit);
                assert_eq!(peer.kv_cache_limit, KvCacheLimit::Context);
            }
        }
        for value in ["0", "-0.8", "2", "NaN", "1e-1", "-0G"] {
            assert!(
                Cli::try_parse_from(["flare", "run", "model.safetensors", "--memory-limit", value])
                    .is_err()
            );
        }
        assert!(
            Cli::try_parse_from([
                "flare",
                "run",
                "model.safetensors",
                "--kv-cache-limit",
                "unknown"
            ])
            .is_err()
        );
    }
}
