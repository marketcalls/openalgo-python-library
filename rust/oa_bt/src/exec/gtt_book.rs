//! The GTTs of a run: every GTT by id, in creation order, never removed, so a result can
//! list them all. Queries return ids in id order, which keeps the slice sequence
//! deterministic.

use crate::model::Product;

use super::error::ExecError;
use super::gtt::{next_gtt_status, Gtt, GttAction, GttId, GttPurpose, GttStatus};
use super::instrument::InstrumentId;
use super::order::OrderId;

/// Every GTT of a run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GttBook {
    gtts: Vec<Gtt>,
}

impl GttBook {
    /// The id the next GTT will get: `g1` for the first.
    pub fn next_id(&self) -> GttId {
        GttId(self.gtts.len() as u64 + 1)
    }

    /// Add a GTT.
    ///
    /// Errors: `InvalidRequest` when its id is not `next_id` (ids are dense and in
    /// creation order).
    pub fn insert(&mut self, gtt: Gtt) -> Result<GttId, ExecError> {
        if gtt.id != self.next_id() {
            return Err(ExecError::InvalidRequest {
                reason: format!("GTT id {} is out of sequence", gtt.id),
            });
        }
        let id = gtt.id;
        self.gtts.push(gtt);
        Ok(id)
    }

    fn slot(id: GttId) -> Option<usize> {
        usize::try_from(id.0).ok()?.checked_sub(1)
    }

    /// The GTT with this id.
    pub fn get(&self, id: GttId) -> Option<&Gtt> {
        self.gtts.get(Self::slot(id)?)
    }

    /// The GTT with this id, for a change of its triggers or quantity.
    pub fn get_mut(&mut self, id: GttId) -> Option<&mut Gtt> {
        let slot = Self::slot(id)?;
        self.gtts.get_mut(slot)
    }

    /// Move a GTT by the state table.
    ///
    /// Errors: `InvalidRequest` for an unknown id, `IllegalGttTransition` when the
    /// table forbids the move.
    pub fn apply(&mut self, id: GttId, action: GttAction) -> Result<GttStatus, ExecError> {
        let gtt = self.get_mut(id).ok_or_else(|| ExecError::InvalidRequest {
            reason: format!("no GTT {id}"),
        })?;
        let status = next_gtt_status(gtt.status, action)?;
        gtt.status = status;
        Ok(status)
    }

    /// Every GTT, in creation order.
    pub fn gtts(&self) -> &[Gtt] {
        &self.gtts
    }

    /// The active GTTs of an instrument, in id order.
    pub fn active_for(&self, instrument: InstrumentId) -> Vec<GttId> {
        self.gtts
            .iter()
            .filter(|gtt| gtt.is_active() && gtt.instrument == instrument)
            .map(|gtt| gtt.id)
            .collect()
    }

    /// The active GTTs that hold the levels of a position, in id order.
    pub fn protecting(&self, instrument: InstrumentId, product: Product) -> Vec<GttId> {
        self.gtts
            .iter()
            .filter(|gtt| {
                gtt.is_active()
                    && gtt.instrument == instrument
                    && gtt.product == product
                    && matches!(gtt.purpose, GttPurpose::Protect(_))
            })
            .map(|gtt| gtt.id)
            .collect()
    }

    /// The active GTT a strategy order rests as, if any.
    pub fn of_order(&self, order: OrderId) -> Option<GttId> {
        self.gtts
            .iter()
            .find(|gtt| gtt.is_active() && gtt.purpose == GttPurpose::Order(order))
            .map(|gtt| gtt.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::gtt::{GttKind, GttLeg, GttLegRole, GttSlot};
    use crate::exec::matching::TriggerDirection;
    use crate::model::Side;

    fn stop_gtt(id: u64, product: Product) -> Gtt {
        Gtt {
            id: GttId(id),
            instrument: InstrumentId(0),
            product,
            side: Side::Sell,
            quantity_units: 10.0,
            kind: GttKind::Single,
            legs: vec![GttLeg {
                role: GttLegRole::Stop,
                trigger_rupees: 1460.0,
                direction: TriggerDirection::Falling,
                child_limit_rupees: 1445.4,
            }],
            purpose: GttPurpose::Protect(GttSlot::Remainder),
            status: GttStatus::Active,
            created_on: 0,
            eligible_from: 0,
            last_modified_slice: None,
            fired: None,
        }
    }

    #[test]
    fn ids_are_dense_and_in_creation_order() {
        let mut book = GttBook::default();
        assert_eq!(book.insert(stop_gtt(1, Product::Cnc)), Ok(GttId(1)));
        assert!(book.insert(stop_gtt(3, Product::Cnc)).is_err());
        assert_eq!(book.next_id(), GttId(2));
    }

    #[test]
    fn a_fired_gtt_no_longer_protects_its_position() {
        let mut book = GttBook::default();
        assert!(book.insert(stop_gtt(1, Product::Cnc)).is_ok());
        assert!(book.insert(stop_gtt(2, Product::Nrml)).is_ok());
        assert_eq!(book.protecting(InstrumentId(0), Product::Cnc), [GttId(1)]);
        assert_eq!(
            book.apply(GttId(1), GttAction::Fire),
            Ok(GttStatus::Triggered)
        );
        assert!(book.protecting(InstrumentId(0), Product::Cnc).is_empty());
        assert_eq!(book.active_for(InstrumentId(0)), [GttId(2)]);
    }
}
