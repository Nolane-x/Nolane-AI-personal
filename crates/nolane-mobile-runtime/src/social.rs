use crate::{
    memory::{MobileMemoryRecord, MobileMemoryStore},
    PersistentMobileState,
    RuntimeError,
};
use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const MOBILE_SOCIAL_OBSERVER_SCHEMA: &str =
    "NOLANE-V060-MOBILE-SOCIAL-PROPOSAL-V1";

const MAX_AFFECT_DELTA: f64 = 0.12;
const MAX_RELATIONSHIP_DELTA: f64 = 0.035;
const MAX_MEMORIES_PER_EVENT: usize = 4;
const MAX_MEMORY_CHARS: usize = 800;
const FACT_CONFIDENCE_FLOOR: f64 = 0.90;

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct MobileMemoryProposal {
    pub text: String,
    pub kind: String,
    pub confidence: f64,
    pub salience: f64,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct MobileSocialProposal {
    pub schema: String,
    #[serde(default)]
    pub affect_delta: BTreeMap<String, f64>,
    #[serde(default)]
    pub relationship_delta: BTreeMap<String, f64>,
    #[serde(default)]
    pub memories: Vec<MobileMemoryProposal>,
    pub uncertainty: f64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct MobileMutationReceipt {
    pub source_event_id: String,
    #[serde(default)]
    pub accepted_affect_delta: BTreeMap<String, f64>,
    #[serde(default)]
    pub accepted_relationship_delta: BTreeMap<String, f64>,
    #[serde(default)]
    pub stored_memory_ids: Vec<String>,
    #[serde(default)]
    pub rejected: Vec<String>,
    pub downgraded_memories: usize,
    pub uncertainty: f64,
}

fn unit(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn bounded(value: f64, limit: f64) -> f64 {
    value.clamp(-limit, limit)
}

fn fresh_memory_id(created_at_ms: u64) -> String {
    let mut bytes = [0u8; 12];
    OsRng.fill_bytes(&mut bytes);
    let suffix = bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("obs-{created_at_ms}-{suffix}")
}

fn preference_tail<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    let folded = text.to_lowercase();
    let index = folded.find(prefix)?;
    let start = index + prefix.len();
    text.get(start..).map(str::trim).filter(|value| !value.is_empty())
}

pub fn deterministic_explicit_preference_observer(
    text: &str,
) -> MobileSocialProposal {
    let trimmed = text.trim();
    let mut proposal = MobileSocialProposal {
        schema: MOBILE_SOCIAL_OBSERVER_SCHEMA.to_string(),
        uncertainty: 0.05,
        ..MobileSocialProposal::default()
    };
    if trimmed.is_empty() {
        return proposal;
    }

    let preference = [
        "tôi thích ",
        "mình thích ",
        "tôi muốn ",
        "mình muốn ",
        "i like ",
        "i prefer ",
        "i want ",
    ]
    .iter()
    .find_map(|prefix| preference_tail(trimmed, prefix));

    if let Some(value) = preference {
        let compact = value
            .trim_matches(|ch: char| matches!(ch, '.' | '!' | '?' | ' '))
            .chars()
            .take(500)
            .collect::<String>();
        if !compact.is_empty() {
            let mut metadata = BTreeMap::new();
            metadata.insert(
                "observer_strategy".into(),
                Value::String("explicit_preference_pattern".into()),
            );
            proposal.memories.push(MobileMemoryProposal {
                text: format!("User preference: {compact}"),
                kind: "preference".into(),
                confidence: 0.95,
                salience: 0.72,
                metadata,
            });
        }
    }
    proposal
}

pub fn apply_social_proposal(
    state: &mut PersistentMobileState,
    memory_store: &mut MobileMemoryStore,
    social_drive: &mut f64,
    proposal: &MobileSocialProposal,
    source_event_id: &str,
    created_at_ms: u64,
) -> Result<MobileMutationReceipt, RuntimeError> {
    if proposal.schema != MOBILE_SOCIAL_OBSERVER_SCHEMA {
        return Err(RuntimeError::Invalid(
            "mobile social proposal schema mismatch".into(),
        ));
    }
    if !proposal.uncertainty.is_finite() {
        return Err(RuntimeError::Invalid(
            "mobile social proposal uncertainty is non-finite".into(),
        ));
    }

    let mut receipt = MobileMutationReceipt {
        source_event_id: source_event_id.to_string(),
        uncertainty: unit(proposal.uncertainty),
        ..MobileMutationReceipt::default()
    };

    for (name, raw) in &proposal.affect_delta {
        if !raw.is_finite() {
            receipt.rejected.push(format!("affect:{name}:non_finite"));
            continue;
        }
        let delta = bounded(*raw, MAX_AFFECT_DELTA);
        match name.as_str() {
            "valence" => {
                state.state.affect.valence =
                    (state.state.affect.valence + delta).clamp(-1.0, 1.0);
            }
            "energy" => {
                state.state.affect.energy =
                    unit(state.state.affect.energy + delta);
            }
            "playfulness" => {
                state.state.affect.playfulness =
                    unit(state.state.affect.playfulness + delta);
            }
            "irritation" => {
                state.state.affect.irritation =
                    unit(state.state.affect.irritation + delta);
            }
            "concern" => {
                state.state.affect.concern =
                    unit(state.state.affect.concern + delta);
            }
            "social_drive" => {
                *social_drive = unit(*social_drive + delta);
            }
            _ => {
                receipt.rejected.push(format!("affect:{name}"));
                continue;
            }
        }
        receipt.accepted_affect_delta.insert(name.clone(), delta);
    }

    for (name, raw) in &proposal.relationship_delta {
        if !raw.is_finite() {
            receipt
                .rejected
                .push(format!("relationship:{name}:non_finite"));
            continue;
        }
        let delta = bounded(*raw, MAX_RELATIONSHIP_DELTA);
        match name.as_str() {
            "closeness" => {
                state.state.relationship.closeness =
                    unit(state.state.relationship.closeness + delta);
            }
            "trust" => {
                state.state.relationship.trust =
                    unit(state.state.relationship.trust + delta);
            }
            "familiarity" => {
                state.state.relationship.familiarity =
                    unit(state.state.relationship.familiarity + delta);
            }
            _ => {
                receipt.rejected.push(format!("relationship:{name}"));
                continue;
            }
        }
        receipt
            .accepted_relationship_delta
            .insert(name.clone(), delta);
    }

    for item in proposal.memories.iter().take(MAX_MEMORIES_PER_EVENT) {
        if !item.confidence.is_finite() || !item.salience.is_finite() {
            receipt.rejected.push("memory:non_finite".into());
            continue;
        }
        let text = item
            .text
            .trim()
            .chars()
            .take(MAX_MEMORY_CHARS)
            .collect::<String>();
        if text.is_empty() {
            receipt.rejected.push("memory:empty".into());
            continue;
        }
        let confidence = unit(item.confidence);
        let mut kind = match item.kind.as_str() {
            "episodic" | "fact" | "preference" | "inference" => {
                item.kind.clone()
            }
            _ => "inference".into(),
        };
        let mut metadata = item.metadata.clone();
        if kind == "fact" && confidence < FACT_CONFIDENCE_FLOOR {
            kind = "inference".into();
            metadata.insert(
                "downgraded_from_fact".into(),
                Value::Bool(true),
            );
            receipt.downgraded_memories += 1;
        }
        metadata.insert(
            "observer_uncertainty".into(),
            Value::from(receipt.uncertainty),
        );
        metadata.insert(
            "proposed_by_observer".into(),
            Value::Bool(true),
        );
        let memory_id = fresh_memory_id(created_at_ms);
        memory_store.push_record(MobileMemoryRecord {
            memory_id: memory_id.clone(),
            text,
            kind,
            salience: unit(item.salience),
            confidence,
            source_event_id: Some(source_event_id.to_string()),
            created_at_ms,
            metadata,
        })?;
        receipt.stored_memory_ids.push(memory_id);
    }

    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ProductPayloadAffect,
        ProductPayloadProfile,
        ProductPayloadRelationship,
        ProductPayloadState,
    };

    fn state() -> PersistentMobileState {
        PersistentMobileState {
            source_checkpoint_sha256: "a".repeat(64),
            latent: vec![0.0; 4],
            profile: ProductPayloadProfile {
                preferred_name: String::new(),
                language: "auto".into(),
                response_length: "balanced".into(),
                conversation_style: "natural".into(),
                personal_instruction: String::new(),
            },
            state: ProductPayloadState {
                identity_id: "identity".into(),
                relationship: ProductPayloadRelationship {
                    closeness: 0.2,
                    trust: 0.2,
                    familiarity: 0.2,
                    interaction_count: 0,
                },
                affect: ProductPayloadAffect {
                    valence: 0.0,
                    energy: 0.5,
                    playfulness: 0.4,
                    concern: 0.0,
                    irritation: 0.0,
                },
                open_threads: Vec::new(),
            },
            memories: Vec::new(),
        }
    }

    #[test]
    fn explicit_preference_is_narrow_and_provenance_bound() {
        let proposal = deterministic_explicit_preference_observer(
            "Tôi thích giao diện màu cam.",
        );
        assert_eq!(proposal.memories.len(), 1);
        assert_eq!(proposal.memories[0].kind, "preference");

        let mut state = state();
        let mut store = MobileMemoryStore::default();
        let mut social_drive = 0.2;
        let receipt = apply_social_proposal(
            &mut state,
            &mut store,
            &mut social_drive,
            &proposal,
            "event-1",
            1000,
        )
        .unwrap();
        assert_eq!(receipt.stored_memory_ids.len(), 1);
        let memory = &store.records[0];
        assert_eq!(memory.source_event_id.as_deref(), Some("event-1"));
        assert_eq!(memory.kind, "preference");
        assert_eq!(
            memory.metadata.get("proposed_by_observer"),
            Some(&Value::Bool(true))
        );
    }

    #[test]
    fn validator_clamps_deltas_and_downgrades_weak_fact() {
        let mut proposal = MobileSocialProposal {
            schema: MOBILE_SOCIAL_OBSERVER_SCHEMA.into(),
            uncertainty: 0.4,
            ..MobileSocialProposal::default()
        };
        proposal.affect_delta.insert("concern".into(), 9.0);
        proposal.relationship_delta.insert("trust".into(), -9.0);
        proposal.memories.push(MobileMemoryProposal {
            text: "Uncertain claim".into(),
            kind: "fact".into(),
            confidence: 0.5,
            salience: 0.8,
            metadata: BTreeMap::new(),
        });

        let mut state = state();
        let mut store = MobileMemoryStore::default();
        let mut social_drive = 0.2;
        let receipt = apply_social_proposal(
            &mut state,
            &mut store,
            &mut social_drive,
            &proposal,
            "event-2",
            2000,
        )
        .unwrap();

        assert_eq!(receipt.accepted_affect_delta["concern"], 0.12);
        assert_eq!(
            receipt.accepted_relationship_delta["trust"],
            -0.035
        );
        assert_eq!(receipt.downgraded_memories, 1);
        assert_eq!(store.records[0].kind, "inference");
    }
}
