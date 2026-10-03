use crate::RuntimeError;
use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    fs::OpenOptions,
    io::Write,
    path::Path,
};

pub const MOBILE_MEMORY_SCHEMA: &str =
    "NOLANE-V060-MOBILE-PROVENANCE-MEMORY-V1";
pub const MAX_MEMORY_RECORDS: usize = 2048;
pub const MAX_MEMORY_LINKS: usize = 8192;
pub const MAX_MEMORY_TEXT_CHARS: usize = 1000;
pub const DEFAULT_RETRIEVAL_LIMIT: usize = 8;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct MobileMemoryRecord {
    pub memory_id: String,
    pub text: String,
    pub kind: String,
    pub salience: f64,
    pub confidence: f64,
    pub source_event_id: Option<String>,
    pub created_at_ms: u64,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct MobileMemoryLink {
    pub parent_memory_id: String,
    pub child_memory_id: String,
    pub relation: String,
    pub source_event_id: String,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct MobileMemoryStore {
    pub records: Vec<MobileMemoryRecord>,
    pub links: Vec<MobileMemoryLink>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct MobileMemoryEnvelope {
    schema: String,
    store_sha256: String,
    store: MobileMemoryStore,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), RuntimeError> {
    let parent = path.parent().ok_or_else(|| RuntimeError::Invalid(
        "mobile memory path has no parent".into(),
    ))?;
    fs::create_dir_all(parent)?;
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| RuntimeError::Invalid(
            "mobile memory path has invalid file name".into(),
        ))?;
    let mut nonce = [0u8; 8];
    OsRng.fill_bytes(&mut nonce);
    let suffix = nonce
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let temporary = parent.join(format!(".{file_name}.{suffix}.tmp"));
    let result = (|| -> Result<(), RuntimeError> {
        let mut handle = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        handle.write_all(bytes)?;
        handle.sync_all()?;
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn canonical_store_value(
    store: &MobileMemoryStore,
) -> Result<Value, RuntimeError> {
    let value = serde_json::to_value(store)?;
    Ok(value)
}

fn store_digest(store: &MobileMemoryStore) -> Result<String, RuntimeError> {
    let value = canonical_store_value(store)?;
    Ok(sha256_hex(&serde_json::to_vec(&value)?))
}

fn valid_kind(kind: &str) -> bool {
    matches!(
        kind,
        "episodic" | "fact" | "preference" | "inference" | "habit"
    )
}

fn lexical_tokens(text: &str) -> HashSet<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| token.chars().count() > 1)
        .map(|token| token.to_lowercase())
        .collect()
}

pub fn lexical_similarity(a: &str, b: &str) -> f64 {
    let left = lexical_tokens(a);
    let right = lexical_tokens(b);
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    left.intersection(&right).count() as f64
        / left.union(&right).count() as f64
}

impl MobileMemoryStore {
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if self.records.len() > MAX_MEMORY_RECORDS {
            return Err(RuntimeError::Invalid(format!(
                "mobile memory record limit exceeded: {} > {}",
                self.records.len(),
                MAX_MEMORY_RECORDS,
            )));
        }
        if self.links.len() > MAX_MEMORY_LINKS {
            return Err(RuntimeError::Invalid(format!(
                "mobile memory link limit exceeded: {} > {}",
                self.links.len(),
                MAX_MEMORY_LINKS,
            )));
        }

        let mut ids = HashSet::new();
        for record in &self.records {
            if record.memory_id.trim().is_empty()
                || !ids.insert(record.memory_id.as_str())
            {
                return Err(RuntimeError::Invalid(
                    "mobile memory IDs must be non-empty and unique".into(),
                ));
            }
            if record.text.trim().is_empty()
                || record.text.chars().count() > MAX_MEMORY_TEXT_CHARS
            {
                return Err(RuntimeError::Invalid(
                    "mobile memory text is empty or too long".into(),
                ));
            }
            if !valid_kind(&record.kind) {
                return Err(RuntimeError::Invalid(
                    "mobile memory kind is invalid".into(),
                ));
            }
            if !record.salience.is_finite()
                || !(0.0..=1.0).contains(&record.salience)
                || !record.confidence.is_finite()
                || !(0.0..=1.0).contains(&record.confidence)
            {
                return Err(RuntimeError::Invalid(
                    "mobile memory confidence/salience is invalid".into(),
                ));
            }
        }

        let ids = self
            .records
            .iter()
            .map(|record| record.memory_id.as_str())
            .collect::<HashSet<_>>();
        for link in &self.links {
            if !ids.contains(link.parent_memory_id.as_str())
                || !ids.contains(link.child_memory_id.as_str())
                || link.relation != "consolidated_into"
                || link.source_event_id.trim().is_empty()
            {
                return Err(RuntimeError::Invalid(
                    "mobile memory provenance link is invalid".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn migrate_legacy_projection(memories: &[String]) -> Self {
        let records = memories
            .iter()
            .enumerate()
            .filter_map(|(index, text)| {
                let clean = text.trim();
                if clean.is_empty() {
                    return None;
                }
                let digest = sha256_hex(
                    format!("{index}\0{clean}").as_bytes(),
                );
                let mut metadata = BTreeMap::new();
                metadata.insert(
                    "migrated_from_v055_projection".into(),
                    Value::Bool(true),
                );
                Some(MobileMemoryRecord {
                    memory_id: format!("legacy-{}", &digest[..24]),
                    text: clean.chars().take(MAX_MEMORY_TEXT_CHARS).collect(),
                    kind: "episodic".into(),
                    salience: 0.5,
                    confidence: 1.0,
                    source_event_id: None,
                    created_at_ms: 0,
                    metadata,
                })
            })
            .collect();
        Self {
            records,
            links: Vec::new(),
        }
    }

    pub fn record_user_message(
        &mut self,
        text: &str,
        source_event_id: &str,
        created_at_ms: u64,
    ) -> Result<String, RuntimeError> {
        if self.records.len() >= MAX_MEMORY_RECORDS {
            return Err(RuntimeError::Invalid(
                "mobile memory store is full".into(),
            ));
        }
        let clean = text.trim();
        if clean.is_empty() {
            return Err(RuntimeError::Invalid(
                "cannot store empty mobile memory".into(),
            ));
        }
        let mut random = [0u8; 12];
        OsRng.fill_bytes(&mut random);
        let random_hex = random
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let memory_id = format!("mem-{created_at_ms}-{random_hex}");
        let length_factor =
            (clean.chars().count() as f64 / 40.0).min(8.0);
        let salience = (0.38 + 0.03 * length_factor).clamp(0.25, 0.9);
        self.records.push(MobileMemoryRecord {
            memory_id: memory_id.clone(),
            text: clean.chars().take(MAX_MEMORY_TEXT_CHARS).collect(),
            kind: "episodic".into(),
            salience,
            confidence: 1.0,
            source_event_id: Some(source_event_id.to_string()),
            created_at_ms,
            metadata: BTreeMap::new(),
        });
        Ok(memory_id)
    }

    pub fn relevant_texts(
        &self,
        query: &str,
        now_ms: u64,
        limit: usize,
    ) -> Vec<String> {
        let query_tokens = lexical_tokens(query);
        let mut scored = self
            .records
            .iter()
            .map(|record| {
                let memory_tokens = lexical_tokens(&record.text);
                let overlap = if query_tokens.is_empty() {
                    0.0
                } else {
                    query_tokens.intersection(&memory_tokens).count() as f64
                        / query_tokens.len().max(1) as f64
                };
                let recency = if record.created_at_ms == 0 {
                    0.0
                } else {
                    let age_days = now_ms
                        .saturating_sub(record.created_at_ms) as f64
                        / 86_400_000.0;
                    (-age_days / 30.0).exp()
                };
                let score =
                    0.52 * overlap + 0.30 * record.salience + 0.18 * recency;
                (score, record)
            })
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    right
                        .1
                        .created_at_ms
                        .cmp(&left.1.created_at_ms)
                })
                .then_with(|| left.1.memory_id.cmp(&right.1.memory_id))
        });
        scored
            .into_iter()
            .take(limit)
            .map(|(_, record)| record.text.clone())
            .collect()
    }

    pub fn unconsolidated_records(&self) -> Vec<&MobileMemoryRecord> {
        let parents = self
            .links
            .iter()
            .map(|link| link.parent_memory_id.as_str())
            .collect::<HashSet<_>>();
        self.records
            .iter()
            .filter(|record| !parents.contains(record.memory_id.as_str()))
            .collect()
    }

    pub fn consolidate_near_duplicates(
        &mut self,
        source_event_id: &str,
        created_at_ms: u64,
        duplicate_similarity: f64,
        max_new_memories: usize,
    ) -> Result<Vec<String>, RuntimeError> {
        let candidates = self
            .unconsolidated_records()
            .into_iter()
            .take(160)
            .cloned()
            .collect::<Vec<_>>();
        let mut used = HashSet::new();
        let mut proposals: Vec<(MobileMemoryRecord, Vec<String>)> = Vec::new();

        for anchor in &candidates {
            if used.contains(&anchor.memory_id) {
                continue;
            }
            let mut cluster = vec![anchor.clone()];
            for other in &candidates {
                if other.memory_id == anchor.memory_id
                    || used.contains(&other.memory_id)
                {
                    continue;
                }
                if lexical_similarity(&anchor.text, &other.text)
                    >= duplicate_similarity
                {
                    cluster.push(other.clone());
                }
                if cluster.len() >= 8 {
                    break;
                }
            }
            if cluster.len() < 2 {
                continue;
            }

            let source_ids = cluster
                .iter()
                .map(|memory| memory.memory_id.clone())
                .collect::<Vec<_>>();
            used.extend(source_ids.iter().cloned());
            let best = cluster
                .iter()
                .max_by(|left, right| {
                    left.confidence
                        .partial_cmp(&right.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| {
                            left.salience
                                .partial_cmp(&right.salience)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .then_with(|| {
                            left.text.len().cmp(&right.text.len())
                        })
                })
                .expect("non-empty consolidation cluster");
            let mut kind = if valid_kind(&best.kind) {
                best.kind.clone()
            } else {
                "inference".into()
            };
            if kind == "fact"
                && cluster.iter().any(|memory| memory.kind != "fact")
            {
                kind = "inference".into();
            }
            let confidence = cluster
                .iter()
                .map(|memory| memory.confidence)
                .fold(1.0_f64, f64::min);
            let salience = (
                cluster
                    .iter()
                    .map(|memory| memory.salience)
                    .fold(0.0_f64, f64::max)
                    + 0.05
            )
            .min(1.0);
            let digest = sha256_hex(
                format!(
                    "{source_event_id}\0{}",
                    source_ids.join("\0")
                )
                .as_bytes(),
            );
            let memory_id = format!(
                "rest-{}-{}",
                created_at_ms,
                &digest[..20],
            );
            let mut metadata = BTreeMap::new();
            metadata.insert("rest_consolidated".into(), Value::Bool(true));
            metadata.insert(
                "source_memory_ids".into(),
                json!(source_ids.clone()),
            );
            metadata.insert(
                "strategy".into(),
                Value::String("near_duplicate_merge".into()),
            );
            metadata.insert(
                "cluster_size".into(),
                json!(cluster.len()),
            );
            proposals.push((
                MobileMemoryRecord {
                    memory_id,
                    text: best.text.clone(),
                    kind,
                    salience,
                    confidence,
                    source_event_id: Some(source_event_id.to_string()),
                    created_at_ms,
                    metadata,
                },
                source_ids,
            ));
            if proposals.len() >= max_new_memories {
                break;
            }
        }

        if self.records.len() + proposals.len() > MAX_MEMORY_RECORDS {
            return Err(RuntimeError::Invalid(
                "mobile memory store is full".into(),
            ));
        }
        if self.links.len()
            + proposals.iter().map(|(_, ids)| ids.len()).sum::<usize>()
            > MAX_MEMORY_LINKS
        {
            return Err(RuntimeError::Invalid(
                "mobile memory provenance link limit exceeded".into(),
            ));
        }

        let mut stored = Vec::new();
        for (record, source_ids) in proposals {
            let child_id = record.memory_id.clone();
            for parent_id in source_ids {
                self.links.push(MobileMemoryLink {
                    parent_memory_id: parent_id,
                    child_memory_id: child_id.clone(),
                    relation: "consolidated_into".into(),
                    source_event_id: source_event_id.to_string(),
                    created_at_ms,
                });
            }
            stored.push(child_id);
            self.records.push(record);
        }
        self.validate()?;
        Ok(stored)
    }

    pub fn projection(&self, now_ms: u64, limit: usize) -> Vec<String> {
        self.relevant_texts("", now_ms, limit)
    }
}

pub fn read_mobile_memory_store(
    path: impl AsRef<Path>,
) -> Result<MobileMemoryStore, RuntimeError> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    let envelope: MobileMemoryEnvelope = serde_json::from_slice(&bytes)?;
    if envelope.schema != MOBILE_MEMORY_SCHEMA {
        return Err(RuntimeError::Invalid(
            "mobile memory store schema mismatch".into(),
        ));
    }
    let actual = store_digest(&envelope.store)?;
    if actual != envelope.store_sha256 {
        return Err(RuntimeError::Invalid(
            "mobile memory store integrity mismatch".into(),
        ));
    }
    envelope.store.validate()?;
    Ok(envelope.store)
}

pub fn write_mobile_memory_store(
    path: impl AsRef<Path>,
    store: &MobileMemoryStore,
) -> Result<(), RuntimeError> {
    store.validate()?;
    let envelope = MobileMemoryEnvelope {
        schema: MOBILE_MEMORY_SCHEMA.to_string(),
        store_sha256: store_digest(store)?,
        store: store.clone(),
    };
    atomic_write(path.as_ref(), &serde_json::to_vec(&envelope)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn legacy_projection_migrates_with_stable_ids() {
        let first = MobileMemoryStore::migrate_legacy_projection(&[
            "Người dùng thích câu trả lời ngắn.".into(),
            "Nolane đang phát triển mobile.".into(),
        ]);
        let second = MobileMemoryStore::migrate_legacy_projection(&[
            "Người dùng thích câu trả lời ngắn.".into(),
            "Nolane đang phát triển mobile.".into(),
        ]);
        assert_eq!(first, second);
        assert!(first.records.iter().all(|memory| {
            memory.metadata
                .get("migrated_from_v055_projection")
                == Some(&Value::Bool(true))
        }));
    }

    #[test]
    fn memory_store_roundtrips_and_rejects_tamper() {
        let root = tempdir().unwrap();
        let path = root.path().join("memory.json");
        let mut store = MobileMemoryStore::default();
        store
            .record_user_message("Tôi thích màu cam.", "event-1", 1_000)
            .unwrap();
        write_mobile_memory_store(&path, &store).unwrap();
        assert_eq!(read_mobile_memory_store(&path).unwrap(), store);

        let mut raw: Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        raw["store"]["records"][0]["text"] =
            Value::String("tampered".into());
        fs::write(&path, serde_json::to_vec(&raw).unwrap()).unwrap();
        let error = read_mobile_memory_store(&path).unwrap_err();
        assert!(error.to_string().contains("integrity mismatch"));
    }

    #[test]
    fn retrieval_matches_overlap_salience_recency_policy() {
        let mut store = MobileMemoryStore::default();
        store.records = vec![
            MobileMemoryRecord {
                memory_id: "a".into(),
                text: "Nolane mobile Android".into(),
                kind: "episodic".into(),
                salience: 0.5,
                confidence: 1.0,
                source_event_id: Some("e1".into()),
                created_at_ms: 1_000,
                metadata: BTreeMap::new(),
            },
            MobileMemoryRecord {
                memory_id: "b".into(),
                text: "một ký ức không liên quan".into(),
                kind: "episodic".into(),
                salience: 0.4,
                confidence: 1.0,
                source_event_id: Some("e2".into()),
                created_at_ms: 9_000,
                metadata: BTreeMap::new(),
            },
        ];
        let ranked =
            store.relevant_texts("Nolane Android", 10_000, 1);
        assert_eq!(ranked, vec!["Nolane mobile Android"]);
    }

    #[test]
    fn consolidation_keeps_sources_and_creates_links() {
        let mut store = MobileMemoryStore::default();
        store
            .record_user_message(
                "Tôi thích màu cam chủ đạo cho Nolane.",
                "event-1",
                1_000,
            )
            .unwrap();
        store
            .record_user_message(
                "Tôi thích màu cam chủ đạo cho Nolane!",
                "event-2",
                2_000,
            )
            .unwrap();
        let source_ids = store
            .records
            .iter()
            .map(|memory| memory.memory_id.clone())
            .collect::<Vec<_>>();
        let created = store
            .consolidate_near_duplicates(
                "rest-1",
                4_000,
                0.72,
                4,
            )
            .unwrap();
        assert_eq!(created.len(), 1);
        assert_eq!(store.records.len(), 3);
        assert_eq!(store.links.len(), 2);
        assert!(source_ids.iter().all(|source| {
            store.records.iter().any(|memory| &memory.memory_id == source)
        }));
        assert!(store.links.iter().all(|link| {
            link.child_memory_id == created[0]
        }));
    }
}
