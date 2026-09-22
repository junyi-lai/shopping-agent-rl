import unittest

from shop_env.slot_lease_pool import SlotLeasePool


class SlotLeasePoolTest(unittest.TestCase):
    def test_terminal_owner_keeps_slot_until_explicit_release(self):
        pool = SlotLeasePool(1)
        slot = pool.acquire()
        self.assertEqual(slot, 0)
        self.assertIsNone(pool.acquire())
        self.assertTrue(pool.release(slot))
        self.assertEqual(pool.acquire(), 0)

    def test_stale_duplicate_release_is_visible(self):
        pool = SlotLeasePool(1)
        slot = pool.acquire()
        self.assertTrue(pool.release(slot))
        self.assertFalse(pool.release(slot))

    def test_reset_recovers_all_slots(self):
        pool = SlotLeasePool(3)
        pool.acquire()
        pool.acquire()
        pool.reset(3)
        self.assertEqual(pool.free_slots(), frozenset({0, 1, 2}))


if __name__ == "__main__":
    unittest.main()
