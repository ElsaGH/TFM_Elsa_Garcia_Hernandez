Simulacrum data release, version 1.1.0, 2018-11-26

This is a package of fake, non-identifiable, simulated cancer registration and
SACT data.

For more Simulacrum documentation and licensing details, please see
  https://simulacrum.healthdatainsight.org.uk/

Any analysis performed on this data will give inaccurate statistical
results. If you would like access to detailed, quality assured statistical
cancer data, please go to https://www.cancerdata.nhs.uk/ or contact the Public
Health England Office for Data Release <ODR@phe.gov.uk>.

To minimise the potential for confusion with real data, organisational codes in
this extract have been obfuscated.

This package contains files to create all the empty table structures (tested on
Oracle 11g), CSV and SQL to populate the lookup tables, the simulated data in
CSV format, Oracle SQLLDR CTL files to import it, and additional documentation:

README.TXT
- This file

create_table_sim_av_patient.sql
create_table_sim_av_tumour.sql
- SQL to create tables SIM_AV_PATIENT and SIM_AV_TUMOUR 
create_tables_sim_sact.sql
- SQL to create tables SIM_SACT_CYCLE, SIM_SACT_DRUG_DETAIL, SIM_SACT_OUTCOME,
  SIM_SACT_PATIENT, SIM_SACT_REGIMEN and SIM_SACT_TUMOUR
create_tables_zlookups.sql
- SQL to create tables ZACE27SCORE, ZALLRED, ZBASIS, ZBEHAVIOUR,
  ZCANCERCAREPLANINTENT, ZCNSLOCATION, ZDEATHLOCATION, ZDIAGNOSTICROUTE,
  ZETHNICITY, ZGRADE, ZHISTOLOGYLOOKUP, ZICD, ZICDCLASSIFICATION, ZLATERALITY,
  ZSEX, ZSTAGE, ZTNMEDITION, ZVITALSTATUS, ZYESNOUNKNOWN

insert_lookups_zace27score.sql
insert_lookups_zallred.sql
insert_lookups_zbasis.sql
insert_lookups_zbehaviour.sql
insert_lookups_zcancercareplanintent.sql
insert_lookups_zcnslocation.sql
insert_lookups_zdeathlocation.sql
insert_lookups_zdiagnosticroute.sql
insert_lookups_zethnicity.sql
insert_lookups_zgrade.sql
insert_lookups_zhistologylookup.sql
insert_lookups_zicd.sql
insert_lookups_zicdclassification.sql
insert_lookups_zlaterality.sql
insert_lookups_zsex.sql
insert_lookups_zstage.sql
insert_lookups_ztnmedition.sql
insert_lookups_zvitalstatus.sql
insert_lookups_zyesnounknown.sql
- Lookup table contents, as SQL INSERTs

sim_av_tumour.ctl
sim_av_patient.ctl
sim_sact_cycle.ctl
sim_sact_drug_detail.ctl
sim_sact_outcome.ctl
sim_sact_patient.ctl
sim_sact_regimen.ctl
sim_sact_tumour.ctl
- Simulated cancer registration and SACT data for 2013-2015, as Oracle SQLLDR
  ctl files, including import scripts.

insert_lookups_zace27score.csv
insert_lookups_zallred.csv
insert_lookups_zbasis.csv
insert_lookups_zbehaviour.csv
insert_lookups_zcancercareplanintent.csv
insert_lookups_zcnslocation.csv
insert_lookups_zdeathlocation.csv
insert_lookups_zdiagnosticroute.csv
insert_lookups_zethnicity.csv
insert_lookups_zgrade.csv
insert_lookups_zhistologylookup.csv
insert_lookups_zicd.csv
insert_lookups_zicdclassification.csv
insert_lookups_zlaterality.csv
insert_lookups_zsex.csv
insert_lookups_zstage.csv
insert_lookups_ztnmedition.csv
insert_lookups_zvitalstatus.csv
insert_lookups_zyesnounknown.csv
- Lookup table contents, as SQL files

sim_av_tumour.csv
sim_av_patient.csv
sim_sact_cycle.csv
sim_sact_drug_detail.csv
sim_sact_outcome.csv
sim_sact_patient.csv
sim_sact_regimen.csv
sim_sact_tumour.csv
- Simulated cancer registration and SACT data for 2013-2015, as CSV files
