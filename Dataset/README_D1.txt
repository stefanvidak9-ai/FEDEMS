README: COFACTOR-Residential Dataset 1

Dataset Overview
This dataset contains hourly sub-metered energy use data from 8 different housing associations in the Oslo area, in total 43 apartment blocks and 2330 apartments collected in the COFACTOR project. The dataset consist of 29 csv-files, one per heating central, meaning that some files contain energy data from only one apartment block, while others contain energy use information from several buildings which are collected together. The CSV-files in the dataset contains meta data about the buildings/heating centrals and hourly time series data of energy use and weather data from the buildings.The energy use data varies between the files, but generally consists of weather data, energy use data from main meters (electricity and district heating) and energy use data from sub-meters (eg. heating).

Dataset accessability
Repository name: data.sintef.no
Data identification number: 
Direct URL to data product: https://data.sintef.no/product/dp-b750d103-a434-45c2-ba5e-1af13f0a866f
Direct URL to dataset: https://data.sintef.no/feature/fe-06016a7f-af61-4208-8598-68c73144b322
Instructions  for accessing the data: The data product is open access and does not require sign-in registration.

Data Summary
- Number of files: 29
- Time series resolution: Hourly
- Time series duration: 2019-2022/2023

Data sources:
Hourly electricity measurements were collected for all available meters, including common areas and individual apartments, from the local distribution system operator (DSO). For privacy reasons, individual apartment electricity measurements are aggregated to total electricity for all apartments served by one heating central. The common electricity meters are labelled based on communication with the building owners and manual interpretation of the data. For the buildings with district heating (DH) measurement data has been collected from the local district heating company.

Files and Structure
The dataset consists of a ZIP-folder with comma separated .txt files. Each file is labelled "Building_XXXX" (XXXX is a number) and contains the building information (meta data) and time series data for an individual building/group of buildings. More information about each building is given in the data descriptor (see citation). 
Each CSV file consists of:
1. Metadata Section
- General building details including ID, construction year, floor area, number of buildings and number of units.
- Heating system details

2. Time Series Data Section
- Timestamp format: %Y-%m-%dT%H:%M:%S%z (local normal time: CET, UTC+01:00)
- Weather columns: outdoor temperature, wind speed, wind direction and global solar radiation
- Energy columns: Main or sub-meters [Wh/h]


Codes Used in CSV Files
Metadata Categories
- Header_line: First line of measurements in the CSV file
- Location: Location of the building
- Year_of_construction: Year the building was constructed
- Floor_area: Floor area in square meters
- Number_of_units: Number of apartments in the building/heating central.
- Number_of_buildings: Number of buildings on the lot
- Building_category: Building type (e.g., Apt for apartment)
- Notes: Additional information about the building and energy data
- Dhw_heat_source: Type of heating technology for domestic hot water
- Sh_heat_source: Type of heating technology for space heating
- ev_chargepoints:Number of chargers and their capacity
- ventilation_types:Number of chargers and their capacity
- number_of_users: Number of users (residents)
- Pv: Photovoltaic system specifications
- Timestamp_format: Format for timestamps in the time series
- Time_zone: Time zone of the dataset (UTC+01:00)
- Building_id: Unique identifier for the building

Metadata Codes:
- Apt: Apartment building category
- CAV_fxd: Constant air volume (fixed)
- EB: Electric boiler
- EFH: Electric floor heater
- EH: Electric heater
- DH: District heating
- GSHP: Ground source heat pump
- HWH: Hot water heater


Weather Columns:
- Tout: Outdoor temperature (Celsius)
- SolGlob: Global Solar Horizontal Radiation (W/m2)
- WindSpd: Wind speed (m/s)
- WindDir: Wind direction (degrees)


Energy Columns:
- ElImp: Imported electricity (Wh/h)
- ElHt: Electricity for electric heaters (Wh/h)
- ElImp_apt:Electricity imported, sum of AMS meters of apartments. (Wh/h) 
- ElOth: Electricity for other purposes (Wh/h)
- ElMix: Electricity for mixed purposes (heating and other) (Wh/h)
- ElPlug: Electricity plug loads (Wh/h)
- ElEV: Electricity used for electric vehicle charging (Wh/h)
- ElAux: Auxiliary electricity use (Wh/h)
- HtDH: Heat from district heating (Wh/h)
- HtSpace: Space heating (Wh/h)
- HtDHW: Domestic hot water heating (Wh/h)
- HtTot: Total heating energy use (Wh/h)


Applications
The dataset can be used for:
- Energy benchmarking and validation of building simulations
- Heating disaggregation research
- Energy time series classification and forecasting
- Grid planning and energy flexibility studies

Limitations
- Sub-meter quality varies, and users should assess data before resampling
- Gaps exist in some time series, though AMS meters and district heating meters are more reliable
- The dataset is specific to Oslo apartment block, which are within the same climate zone.

Citation
If you use this dataset, please cite:
Sørensen, Å. L., Lien, S. K., & Walnum, H. T. (n.d.). COFACTOR-Residential: Hourly electricity and heating data from residential buildings in Norway. Manuscript submitted for publication in Data in Brief.

Contacts
For questions about the dataset, contact:
Synne Krekling Lien (SINTEF Community)
Email: synne.lien@sintef.no
Harald Taxt Walnum (SINTEF Community)
Email:harald.walnum@sintef.no

Acknowledgements
Dataset 1 is published with permission and help from Dråpe Entreprenør AS, Elvia (DSO), and Sikom Connect AS and we greatly acknowledge them for sharing the access to the data. The dataset is provided by the projects “Coincidence factors and peak loads of buildings in the Norwegian low carbon society” (COFACTOR) funded by Research Council of Norway and partners through grant number 326891. The authors gratefully acknowledge the support from the Research Council of Norway  and research partners.

License
This dataset is licensed under Creative Commons Attribution 4.0 (CC BY 4.0).