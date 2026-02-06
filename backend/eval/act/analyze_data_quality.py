"""Analyze KQC database for real vs test data."""
import sys

sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv

load_dotenv()

from backend.act_fetch import _get, get_database

# Test indicators - names containing these are likely test records
TEST_INDICATORS = ["test", "123", "demo", "sample", "fake", "dummy", "xxx", "aaa", "asdf"]

def is_test_record(name: str) -> bool:
    """Check if a name indicates test data."""
    if not name:
        return False
    name_lower = name.lower()
    return any(ind in name_lower for ind in TEST_INDICATORS)

def analyze_opportunities():
    """Analyze opportunity data quality."""
    print("=" * 70)
    print("OPPORTUNITIES ANALYSIS")
    print("=" * 70)

    # Get ALL opportunities without status filter
    opps = _get('/api/opportunities', {'$top': 200})
    print(f"Total opportunities (no filter): {len(opps)}")

    # Check status distribution
    status_dist = {}
    for o in opps:
        s = o.get('status')
        status_dist[s] = status_dist.get(s, 0) + 1
    print(f"\nStatus distribution: {status_dist}")
    print("  (0=Open, 1=Won, 2=Lost typically)")

    # Also try the API with explicit filter
    print("\nTrying with explicit status=0 filter...")
    open_opps_filtered = _get('/api/opportunities', {'$filter': 'status eq 0', '$top': 200})
    print(f"Open opportunities (status eq 0): {len(open_opps_filtered)}")

    # Separate real vs test from open opps
    test_opps = [o for o in open_opps_filtered if is_test_record(o.get('name', ''))]
    real_opps = [o for o in open_opps_filtered if not is_test_record(o.get('name', ''))]

    print(f"  Test records: {len(test_opps)}")
    print(f"  Real records: {len(real_opps)}")

    if real_opps:
        print(f"\nReal OPEN opportunity data quality ({len(real_opps)} records):")
        has_value = sum(1 for o in real_opps if (o.get('productTotal') or 0) > 0)
        has_contact = sum(1 for o in real_opps if o.get('contacts'))
        has_company = sum(1 for o in real_opps if o.get('companies'))
        has_stage = sum(1 for o in real_opps if o.get('stage'))
        has_close_date = sum(1 for o in real_opps if o.get('estimatedCloseDate') and '9999' not in str(o.get('estimatedCloseDate')))
        has_probability = sum(1 for o in real_opps if (o.get('probability') or 0) > 0)
        has_days_in_stage = sum(1 for o in real_opps if (o.get('daysInStage') or 0) > 0)

        print(f"  Has value (productTotal > 0): {has_value}/{len(real_opps)} ({100*has_value/len(real_opps):.1f}%)")
        print(f"  Has contact: {has_contact}/{len(real_opps)} ({100*has_contact/len(real_opps):.1f}%)")
        print(f"  Has company: {has_company}/{len(real_opps)} ({100*has_company/len(real_opps):.1f}%)")
        print(f"  Has stage: {has_stage}/{len(real_opps)} ({100*has_stage/len(real_opps):.1f}%)")
        print(f"  Has close date: {has_close_date}/{len(real_opps)} ({100*has_close_date/len(real_opps):.1f}%)")
        print(f"  Has probability: {has_probability}/{len(real_opps)} ({100*has_probability/len(real_opps):.1f}%)")
        print(f"  Has daysInStage: {has_days_in_stage}/{len(real_opps)} ({100*has_days_in_stage/len(real_opps):.1f}%)")

        # Value distribution
        values = [(o.get('productTotal') or 0) for o in real_opps]
        values_nonzero = [v for v in values if v > 0]
        if values_nonzero:
            print("\n  Value stats (non-zero only):")
            print(f"    Min: ${min(values_nonzero):,.2f}")
            print(f"    Max: ${max(values_nonzero):,.2f}")
            print(f"    Avg: ${sum(values_nonzero)/len(values_nonzero):,.2f}")
            print(f"    Total pipeline: ${sum(values_nonzero):,.2f}")

        # Show sample names
        print("\n  Sample open opportunity names:")
        for o in real_opps[:10]:
            val = o.get('productTotal') or 0
            stage = o.get('stage', {})
            stage_name = stage.get('name') if isinstance(stage, dict) else str(stage)
            print(f"    - {o.get('name')[:50]} | ${val:,.0f} | {stage_name}")

    # Show test record names
    if test_opps:
        print(f"\nTest record names found ({len(test_opps)}):")
        for o in test_opps[:5]:
            print(f"    - {o.get('name')}")

    return real_opps, test_opps

def analyze_contacts():
    """Analyze contact data quality."""
    print("\n" + "=" * 70)
    print("CONTACTS ANALYSIS")
    print("=" * 70)

    contacts = _get('/api/contacts', {'$top': 200})
    print(f"Total contacts: {len(contacts)}")

    # Separate real vs test
    test_contacts = [c for c in contacts if is_test_record(c.get('fullName', '') or c.get('firstName', ''))]
    real_contacts = [c for c in contacts if not is_test_record(c.get('fullName', '') or c.get('firstName', ''))]

    print(f"  Test records: {len(test_contacts)} ({100*len(test_contacts)/len(contacts):.1f}%)")
    print(f"  Real records: {len(real_contacts)} ({100*len(real_contacts)/len(contacts):.1f}%)")

    # Data quality
    if real_contacts:
        print(f"\nReal contact data quality ({len(real_contacts)} records):")
        has_email = sum(1 for c in real_contacts if c.get('emailAddress'))
        has_phone = sum(1 for c in real_contacts if c.get('businessPhone') or c.get('mobilePhone'))
        has_company = sum(1 for c in real_contacts if c.get('company') or c.get('companyID'))
        has_last_reach = sum(1 for c in real_contacts if c.get('lastReach'))

        print(f"  Has email: {has_email}/{len(real_contacts)} ({100*has_email/len(real_contacts):.1f}%)")
        print(f"  Has phone: {has_phone}/{len(real_contacts)} ({100*has_phone/len(real_contacts):.1f}%)")
        print(f"  Has company: {has_company}/{len(real_contacts)} ({100*has_company/len(real_contacts):.1f}%)")
        print(f"  Has lastReach: {has_last_reach}/{len(real_contacts)} ({100*has_last_reach/len(real_contacts):.1f}%)")

    return real_contacts, test_contacts

def analyze_companies():
    """Analyze company data quality."""
    print("\n" + "=" * 70)
    print("COMPANIES ANALYSIS")
    print("=" * 70)

    companies = _get('/api/companies', {'$top': 200})
    print(f"Total companies: {len(companies)}")

    # Separate real vs test
    test_companies = [c for c in companies if is_test_record(c.get('name', ''))]
    real_companies = [c for c in companies if not is_test_record(c.get('name', ''))]

    print(f"  Test records: {len(test_companies)} ({100*len(test_companies)/len(companies):.1f}%)")
    print(f"  Real records: {len(real_companies)} ({100*len(real_companies)/len(companies):.1f}%)")

    return real_companies, test_companies

def analyze_history():
    """Analyze history/activity data."""
    print("\n" + "=" * 70)
    print("HISTORY/ACTIVITIES ANALYSIS")
    print("=" * 70)

    history = _get('/api/history', {'$orderby': 'startTime desc', '$top': 100})
    print(f"Total history items: {len(history)}")

    # Recent activity
    from datetime import datetime, timedelta
    recent_cutoff = (datetime.now() - timedelta(days=30)).isoformat()
    recent = [h for h in history if (h.get('startTime') or '') > recent_cutoff]
    print(f"Recent (last 30 days): {len(recent)}/{len(history)}")

    return history

def main():
    print(f"DATABASE: {get_database()}")
    print("=" * 70)
    print("KQC DATABASE DATA QUALITY ANALYSIS")
    print("=" * 70)

    real_opps, test_opps = analyze_opportunities()
    real_contacts, test_contacts = analyze_contacts()
    real_companies, test_companies = analyze_companies()
    history = analyze_history()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - QUESTION VIABILITY")
    print("=" * 70)

    opps_with_value = [o for o in real_opps if (o.get('productTotal') or 0) > 0]
    opps_with_contacts = [o for o in real_opps if o.get('contacts')]
    recent_history = len(history)

    print(f"""
Data Summary:
- Real open opportunities: {len(real_opps)}
- With value > $0: {len(opps_with_value)}
- With contacts: {len(opps_with_contacts)}
- Real contacts: {len(real_contacts)}
- Real companies: {len(real_companies)}
- Recent activities: {recent_history}

Question Viability (based on data availability):
1. Daily briefing: {'OK' if recent_history > 10 else 'POOR'} - needs history + calendar
2. Pipeline forecast: {'OK' if len(opps_with_value) > 5 else 'POOR'} - needs opps with value/dates
3. Account expansion: {'POOR' if len(real_companies) < 10 else 'OK'} - needs company data ({len(real_companies)} companies)
4. At-risk deals: {'OK' if len(real_opps) > 5 else 'POOR'} - needs open opps with stage/time
5. Relationship gaps: {'OK' if len(opps_with_contacts) > 5 else 'POOR'} - needs opps linked to contacts
6. Top opportunities: {'OK' if len(opps_with_value) > 5 else 'POOR'} - needs opps with value
7. Deals stuck in stage: {'OK' if len(real_opps) > 5 else 'POOR'} - needs opps with daysInStage
""")

if __name__ == "__main__":
    main()
