1. Metadata Layer (Critical)

Every entity should have metadata.

UUID
Name
Alias
Description
Status
Version
Revision
Created By
Updated By
Timestamp
Source
Confidence
Owner
Access Level
Tags
Language
Retention Policy
Validity
Approval Status
Digital Signature
2. Provenance (Very Important)

Instead of storing only facts, store where every fact came from.

Source Document
Page Number
Paragraph
Bounding Box
OCR Confidence
Extraction Method
Model Version
Prompt Version
Human Reviewer
Review Date
Evidence Chain
Citation

Example

Pressure = 7.2 bar

Source:
Manual.pdf

Page:
43

Paragraph:
2

Confidence:
98%

Verified by:
Engineer

Without provenance, users won't trust AI.

3. Temporal Knowledge

Everything changes over time.

Installed
Commissioned
Modified
Repaired
Calibrated
Retired
Archived
Superseded
Expired
Effective Date
Validity Window

Relationship

Pump A

was Motor X

until 2024

then

Motor Y
4. Spatial Knowledge

Industry is spatial.

GPS

Building

Floor

Room

Area

Bay

Rack

Section

Elevation

Coordinates

Pipeline Segment

GIS Region

Relationships

Pump

inside

Room

Room

inside

Building

Building

inside

Plant
5. Sensor & IoT Layer

This becomes the live brain.

Sensor

Tag

Telemetry

Reading

Sampling Rate

Alarm Limit

Current Value

Historical Trend

Signal Quality

Data Gap

Anomaly

Forecast
6. Event Layer

Everything important is an event.

Startup

Shutdown

Trip

Failure

Maintenance

Inspection

Alarm

Calibration

Audit

Power Loss

Network Failure

Emergency

Events create timelines.

7. Workflow Layer
Request

Approval

Assignment

Task

Checklist

Review

Escalation

Closure

Reopen

Verification

Relationships

Task

assigned_to

Engineer

Review

approved_by

Manager
8. Financial Layer

Often forgotten.

Cost

Budget

Purchase

Vendor

Invoice

Warranty

AMC

Maintenance Cost

Downtime Cost

Energy Cost

ROI

Penalty
9. Supply Chain
Supplier

Warehouse

Inventory

Purchase Order

Delivery

Shipment

Stock

Batch

Lot

Lead Time
10. Manufacturing Layer
Product

SKU

Recipe

BOM

Operation

Station

Cycle Time

Yield

Scrap

OEE

Batch

Lot
11. AI Reasoning Layer

This is what makes it intelligent.

Fact

Observation

Hypothesis

Rule

Constraint

Inference

Recommendation

Prediction

Explanation

Alternative

Counterexample

Confidence

Evidence

Goal

Decision
12. Knowledge Evolution

Knowledge changes.

Draft

Reviewed

Approved

Deprecated

Archived

Superseded

Version

Revision

Branch

Merge
13. Communication Layer
Email

Teams Chat

WhatsApp

Call

Meeting

Minutes

Voice Note

Transcript

Discussion

Comment

Decision

A lot of tacit knowledge hides here.

14. Semantic Layer

AI understands aliases.

Pump

==

Transfer Pump

==

Feed Pump

==

P-101

==

Pump-1

Need

Synonym

Alias

Abbreviation

Translation

Canonical Name
15. Taxonomy Layer
Equipment Type

Failure Type

Hazard Type

Maintenance Type

Document Type

Inspection Type

Risk Type
16. Rules Engine
IF

Pressure > 8

AND

Temperature > 70

THEN

Risk = High
17. Causal Graph
Leak

↓

Pressure Drop

↓

Pump Cavitation

↓

Bearing Failure

↓

Motor Trip

↓

Shutdown

This powers RCA.

18. Dependency Graph
Pump

depends on

Motor

Motor

depends on

Power

Power

depends on

Transformer

Transformer

depends on

Substation

Useful for impact analysis.

19. Risk Graph
Risk

Likelihood

Impact

Severity

Detection

Mitigation

Residual Risk
20. External Knowledge

Connect internal knowledge to external sources.

OEM Manuals

ISO Standards

Research Papers

Vendor Bulletins

Government Notifications

Safety Alerts

Failure Databases

Best Practices
21. Learning Layer
Incident

↓

Lesson Learned

↓

Recommendation

↓

Training

↓

Knowledge Card

↓

Quiz

↓

Certification
22. Multi-modal Layer

Don't store only text.

Images

Videos

Audio

Drone Inspection

Thermal Images

CAD

3D Models

P&ID

Blueprints

PDF

Scans
23. Graph Analytics

Derived—not directly stored.

Centrality

Community

Similarity

Shortest Path

Clusters

Knowledge Gap

Influence Score

Critical Asset

Critical Expert

Critical Document
24. Trust Layer

Every answer should include:

Confidence

Evidence

Supporting Documents

Conflicting Documents

Reviewer

Verification Status

Last Updated
25. Tacit Knowledge Layer (the real differentiator)

Capture what never appears in manuals:

Expert heuristics ("Compressor B always needs a warm-up in winter.")
Failure signatures ("A whining sound usually precedes bearing failure by 2–3 days.")
Seasonal behavior ("Cooling efficiency drops during monsoon due to humidity.")
Workarounds ("Restarting PLC before resetting the VFD avoids false alarms.")
Decision rationale ("We chose Vendor X because their seals lasted 30% longer in corrosive service.")
Lessons learned from incidents and near misses.
Common misconceptions and myths.
Tribal knowledge unique to a specific plant or team.