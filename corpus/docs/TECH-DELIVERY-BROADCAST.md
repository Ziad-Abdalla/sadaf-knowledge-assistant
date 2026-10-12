---
doc_id: TECH-DELIVERY-BROADCAST
title: Broadcast Delivery Specification (TV Channel)
department: technical
owner: post-production@sadaf.example
doc_type: spec
version: "3.0"
effective_date: 2026-01-01
supersedes: null
access: [all]
confidentiality: internal
language: en
odoo_link: null
synthetic: true
---
# Broadcast Delivery Specification (TV Channel)
> Synthetic demo document, not a real company document.

## 1. Scope
1.1 This specification defines the technical requirements for delivering finished episodes to linear TV channels.
1.2 Streaming platform deliveries follow a separate specification (TECH-DELIVERY-STREAM).
1.3 Post-production is responsible for QC against this specification before any delivery.

## 2. Video
2.1 Codec: Apple ProRes 422 HQ.
2.2 Resolution: 1920x1080.
2.3 Frame rate: 25 fps progressive.
2.4 Each file contains a single episode with no bars, tone or slate inside the programme content, unless the channel requests otherwise in writing.

## 3. Audio
3.1 Two mixes are delivered: stereo and 5.1.
3.2 Loudness: -23 LUFS integrated, measured per EBU R128.
3.3 True peak: maximum -1 dBTP.
3.4 Loudness reports for both mixes are attached to each delivery.

## 4. Subtitles
4.1 Subtitles are delivered in Arabic and English as SRT files.
4.2 Maximum 42 characters per line.
4.3 Subtitle files are synchronised to the delivered master and named to match it.

## 5. Delivery Schedule and Method
5.1 Episodes are delivered 21 days before the air date.
5.2 Delivery is via the channel's Aspera server, using credentials issued by the channel to post-production.

## 6. File Naming
6.1 Files are named SERIES_EP###_VER.mov, for example RIMAL_EP007_V2.mov.
6.2 The version number increases with every redelivery of the same episode.
