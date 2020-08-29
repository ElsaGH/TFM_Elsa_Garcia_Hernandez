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

This package contains the following additional files:

README.TXT
- This file

Simulacrum-data-dictionary_TFM.xlsx
CPRD Cancer Registration Dictionary Set 17 up to 2016 v8.1
NHS codes.pdf
- This is the data dictionary of the database, as well as two other sources which contain further information about the classification of some variables

sim_av_tumour.csv
sim_av_patient.csv
sim_sact_patient.csv
sim_sact_regimen.csv
sim_sact_tumour.csv
- Simulated cancer registration and SACT data for 2013-2015, as CSV files. Of note, the tables sim_sact_cycle, sim_sact_drug_details and sim_sact_outome that were part of the data downloaded from the source have not been included in this folder since these were not used for this project.

Within the lookup_tables.zip folder:
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


