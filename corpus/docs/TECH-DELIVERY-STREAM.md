---
doc_id: TECH-DELIVERY-STREAM
title: Streaming Platform Delivery Specification
department: technical
owner: post-production@sadaf.example
doc_type: spec
version: "2.0"
effective_date: 2026-02-01
supersedes: null
access: [all]
confidentiality: internal
language: en
odoo_link: null
synthetic: true
---
# Streaming Platform Delivery Specification
> Synthetic demo document, not a real company document.

## 1. Scope
1.1 This specification defines the technical requirements for delivering finished episodes to streaming platforms.
1.2 Linear TV deliveries follow the Broadcast Delivery Specification (TECH-DELIVERY-BROADCAST).
1.3 Where a platform issues its own written specification for a title, that specification prevails for that title.

## 2. Video
2.1 Package: IMF App 2E (IMF Application 2E).
2.2 Resolution: 3840x2160 (UHD).
2.3 Frame rate: 25 fps.
2.4 HDR10 is optional; when an HDR10 version is delivered, an SDR version is delivered alongside it in the same IMF package.

## 3. Audio
3.1 Two mixes are delivered: 5.1 and stereo.
3.2 Loudness: -24 LKFS, dialogue-gated.
3.3 True peak: maximum -2 dBTP.
3.4 Loudness reports for both mixes are attached to each delivery.

## 4. Subtitles
4.1 Subtitles are delivered as timed text in IMSC1 format.
4.2 Languages: Arabic and English.
4.3 Subtitle files are included in the IMF package as separate tracks.

## 5. Delivery Schedule
5.1 Episodes are delivered 30 days before the release date.
5.2 Delivery method and upload credentials are agreed with each platform per title.

## 6. Quality Control
6.1 Post-production runs a full QC pass on every package before upload, and keeps the QC report on file.
6.2 Any platform rejection is logged and redelivered with an incremented version.
