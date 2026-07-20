Physical World
        │
        ▼
Digital Representation
        │
        ▼
Knowledge Graph
        │
        ▼
Reasoning + AI Agents
        │
        ▼
Actions / Predictions / Insights
Level 1 — Master Entity Categories

Instead of 20 entities, industrial intelligence usually ends up with 120-200 node types.

Here's a practical ontology.

1. Organization (15)
Organization
Business Unit
Plant
Site
Department
Division
Workshop
Production Line
Area
Zone
Building
Floor
Room
Vendor
Customer

Relationships

Plant -> has Area

Area -> has Production Line

Department -> owns Equipment

Vendor -> supplies Equipment

Customer -> receives Product
2. Human Knowledge (20)
Employee
Operator
Engineer
Maintenance Engineer
Technician
Supervisor
Manager
Safety Officer
Inspector
Auditor
Contractor
Expert
Trainer
Shift
Role
Skill
Certification
Training Record
Experience
Knowledge Note

Relationships

Engineer -> authored SOP

Operator -> operates Pump

Expert -> solved Incident

Employee -> belongs Department

Employee -> has Skill

Employee -> works Shift

Expert -> mentored Employee

Tacit knowledge

John always starts Pump-7 manually during winter.

Valve-12 vibrates before failure.

Temperature sensor is unreliable after rain.

Motor takes 15 min to stabilize.

Don't trust pressure reading immediately after startup.

This knowledge exists nowhere officially.

3. Asset Hierarchy (30)
Asset

Equipment

Machine

Pump

Motor

Valve

Compressor

Heat Exchanger

Tank

Boiler

Generator

Transformer

Bearing

Seal

Gearbox

Conveyor

Sensor

Actuator

PLC

DCS

SCADA Node

Panel

Pipe

Pipeline

Instrument

Cable

Foundation

Spare Part

Assembly

Subassembly

Relationships

Pump
    has Motor

Motor
    has Bearing

Bearing
    replaced by Spare Part

Valve
    connected_to Pipeline

Pipeline
    belongs Plant

Sensor
    measures Pressure
4. Documents (25)
PDF
SOP
Manual
OEM Manual
Drawing
P&ID
Isometric Drawing
Maintenance Record
Inspection Report
Permit
Checklist
Calibration Report
Work Order
Purchase Order
Invoice
Email
Memo
Audit Report
Incident Report
Near Miss
Root Cause Analysis
Risk Assessment
Standard
Specification
MSDS

Relationships

SOP
describes Pump

Inspection Report
inspected Valve

Drawing
references Tank

Email
mentions Work Order

Manual
belongs Equipment

Audit
found Issue
5. Operations (20)
Operation
Startup
Shutdown
Batch
Production Run
Recipe
Shift Log
Alarm
Event
Trip
Fault
Maintenance
Breakdown
Inspection
Calibration
Cleaning
Repair
Replacement
Testing
Commissioning

Relationships

Maintenance
fixed Pump

Trip
caused Shutdown

Alarm
occurred during Shift

Calibration
performed on Sensor

Repair
used Spare Part
6. Process Engineering (20)
Process

Unit Operation

Flow

Pressure

Temperature

Level

Density

Viscosity

Flow Rate

Setpoint

Operating Window

Control Loop

PID Controller

Feed

Product

Intermediate

Raw Material

Chemical

Catalyst

Reaction

Relationships

Pump

controls Flow

Sensor

measures Temperature

PID

controls Valve

Reaction

requires Catalyst
7. Quality (20)
Quality Check
Inspection
Defect
Deviation
CAPA
NCR
Specification
Tolerance
Batch
Sample
Result
Measurement
Test
Certificate
Acceptance Criteria
Rejection
Trend
Complaint
Audit Finding
Observation

Relationships

Inspection
found Defect

Defect
triggered CAPA

Sample
belongs Batch

Complaint
linked Product
8. Safety (20)
Hazard

Risk

Incident

Near Miss

LOTO

PPE

Permit

Fire

Explosion

Leak

Gas Release

Confined Space

Hot Work

Emergency

Evacuation

Risk Matrix

Barrier

Safety Observation

Unsafe Act

Unsafe Condition

Relationships

Leak
caused Incident

Hazard
located Area

Incident
involved Employee

Permit
required Hot Work
9. Maintenance (25)
Preventive Maintenance
Predictive Maintenance
Corrective Maintenance
Failure
Failure Mode
Failure Cause
Failure Effect
FMEA
RCA
MTBF
MTTR
Spare
Inventory
Tool
Lubricant
Maintenance Plan
Task
Inspection Route
Schedule
Downtime
Uptime
Cost
Labor
Vendor Visit
Warranty

Relationships

Failure
caused Downtime

Maintenance Plan
includes Task

Bearing
failed due Lubrication

Failure
identified RCA
10. Regulatory (20)
ISO Standard
OISD
Factory Act
PESO
Pollution Board
Audit
Compliance Requirement
Violation
Evidence
Inspection
Permit
Certificate
Emission
Waste
Water
Noise
Environmental Limit
Fine
Corrective Action
Review

Relationships

Requirement
requires Evidence

Audit
verified Compliance

Violation
triggered CAPA
11. Time (15)
Date

Time

Shift

Week

Month

Quarter

Year

Maintenance Cycle

Inspection Cycle

Calibration Cycle

Event Timeline

History

Version

Revision

Lifecycle Stage
12. AI Generated Knowledge (20)
Embedding

Chunk

Summary

Keyword

Entity

Relationship

Observation

Insight

Recommendation

Prediction

Risk Score

Confidence

Evidence

Source

Hypothesis

Pattern

Similarity Cluster

Root Cause

Anomaly

Knowledge Card

Relationships

Observation

supported by Documents

Recommendation

based on Incident

Prediction

uses Sensor Data

Knowledge Card

links Equipment
Hidden Tacit Knowledge Layer (Most Important)

This is what retirees know.

Equipment heuristics
Pump vibrates every Monday after restart.

Valve 17 sticks during winter.

Motor B overheats if ambient >42°C.

Tank level sensor drifts after cleaning.

Heat exchanger fouls faster with supplier X.
Human heuristics
Operator A can identify bearing failure by sound.

Senior engineer knows compressor restart sequence.

Night shift uses unofficial startup checklist.

Maintenance team always replaces gasket together with seal.
Process heuristics
Pressure spikes before catalyst degradation.

Current increases 3 days before bearing failure.

Flow oscillation predicts valve blockage.

Temperature rise predicts leakage.
Maintenance heuristics
Bearing usually fails after seal leak.

If vibration >7 mm/s
AND
temperature >85°C

replace bearing.
Supply chain heuristics
Vendor A bearings last 18 months.

Vendor B seals fail in humid season.

OEM parts reduce downtime by 40%.
Regulatory heuristics
Every pressure vessel inspection
requires

Calibration report

Previous inspection

Certificate

Inspector approval

Photo evidence
Relationship Types (Core)

Instead of just

Equipment -> Document

Use semantic edges.

owns

located_at

contains

connected_to

feeds

controls

monitors

operates

maintains

inspected

reported

caused

affected

triggered

resolved

references

depends_on

requires

uses

supplies

installed_at

manufactured_by

calibrated_by

replaced_by

approved_by

verified_by

violates

complies_with

derived_from

mentions

similar_to

version_of

precedes

follows

observed_by

predicted_by

explains

supports

invalidates

Around 60–100 edge types is typical for an industrial knowledge graph.

Final Scale

A production-grade Industrial Knowledge Intelligence platform typically models approximately:

Category	Approx. entities
Organization	15
Human	20
Assets	30
Documents	25
Operations	20
Process	20
Quality	20
Safety	20
Maintenance	25
Regulatory	20
Time	15
AI Knowledge	20
Total	250+ entity types