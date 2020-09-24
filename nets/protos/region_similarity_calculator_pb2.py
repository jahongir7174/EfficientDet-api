# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nets/protos/region_similarity_calculator.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='nets/protos/region_similarity_calculator.proto',
  package='nets.protos',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=b'\n.nets/protos/region_similarity_calculator.proto\x12\x0bnets.protos\"\xae\x02\n\x1aRegionSimilarityCalculator\x12\x42\n\x16neg_sq_dist_similarity\x18\x01 \x01(\x0b\x32 .nets.protos.NegSqDistSimilarityH\x00\x12\x34\n\x0eiou_similarity\x18\x02 \x01(\x0b\x32\x1a.nets.protos.IouSimilarityH\x00\x12\x34\n\x0eioa_similarity\x18\x03 \x01(\x0b\x32\x1a.nets.protos.IoaSimilarityH\x00\x12K\n\x1athresholded_iou_similarity\x18\x04 \x01(\x0b\x32%.nets.protos.ThresholdedIouSimilarityH\x00\x42\x13\n\x11region_similarity\"\x15\n\x13NegSqDistSimilarity\"\x0f\n\rIouSimilarity\"\x0f\n\rIoaSimilarity\"6\n\x18ThresholdedIouSimilarity\x12\x1a\n\riou_threshold\x18\x01 \x01(\x02:\x03\x30.5'
)




_REGIONSIMILARITYCALCULATOR = _descriptor.Descriptor(
  name='RegionSimilarityCalculator',
  full_name='nets.protos.RegionSimilarityCalculator',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='neg_sq_dist_similarity', full_name='nets.protos.RegionSimilarityCalculator.neg_sq_dist_similarity', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='iou_similarity', full_name='nets.protos.RegionSimilarityCalculator.iou_similarity', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ioa_similarity', full_name='nets.protos.RegionSimilarityCalculator.ioa_similarity', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='thresholded_iou_similarity', full_name='nets.protos.RegionSimilarityCalculator.thresholded_iou_similarity', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='region_similarity', full_name='nets.protos.RegionSimilarityCalculator.region_similarity',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=64,
  serialized_end=366,
)


_NEGSQDISTSIMILARITY = _descriptor.Descriptor(
  name='NegSqDistSimilarity',
  full_name='nets.protos.NegSqDistSimilarity',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=368,
  serialized_end=389,
)


_IOUSIMILARITY = _descriptor.Descriptor(
  name='IouSimilarity',
  full_name='nets.protos.IouSimilarity',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=391,
  serialized_end=406,
)


_IOASIMILARITY = _descriptor.Descriptor(
  name='IoaSimilarity',
  full_name='nets.protos.IoaSimilarity',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=408,
  serialized_end=423,
)


_THRESHOLDEDIOUSIMILARITY = _descriptor.Descriptor(
  name='ThresholdedIouSimilarity',
  full_name='nets.protos.ThresholdedIouSimilarity',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='iou_threshold', full_name='nets.protos.ThresholdedIouSimilarity.iou_threshold', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=425,
  serialized_end=479,
)

_REGIONSIMILARITYCALCULATOR.fields_by_name['neg_sq_dist_similarity'].message_type = _NEGSQDISTSIMILARITY
_REGIONSIMILARITYCALCULATOR.fields_by_name['iou_similarity'].message_type = _IOUSIMILARITY
_REGIONSIMILARITYCALCULATOR.fields_by_name['ioa_similarity'].message_type = _IOASIMILARITY
_REGIONSIMILARITYCALCULATOR.fields_by_name['thresholded_iou_similarity'].message_type = _THRESHOLDEDIOUSIMILARITY
_REGIONSIMILARITYCALCULATOR.oneofs_by_name['region_similarity'].fields.append(
  _REGIONSIMILARITYCALCULATOR.fields_by_name['neg_sq_dist_similarity'])
_REGIONSIMILARITYCALCULATOR.fields_by_name['neg_sq_dist_similarity'].containing_oneof = _REGIONSIMILARITYCALCULATOR.oneofs_by_name['region_similarity']
_REGIONSIMILARITYCALCULATOR.oneofs_by_name['region_similarity'].fields.append(
  _REGIONSIMILARITYCALCULATOR.fields_by_name['iou_similarity'])
_REGIONSIMILARITYCALCULATOR.fields_by_name['iou_similarity'].containing_oneof = _REGIONSIMILARITYCALCULATOR.oneofs_by_name['region_similarity']
_REGIONSIMILARITYCALCULATOR.oneofs_by_name['region_similarity'].fields.append(
  _REGIONSIMILARITYCALCULATOR.fields_by_name['ioa_similarity'])
_REGIONSIMILARITYCALCULATOR.fields_by_name['ioa_similarity'].containing_oneof = _REGIONSIMILARITYCALCULATOR.oneofs_by_name['region_similarity']
_REGIONSIMILARITYCALCULATOR.oneofs_by_name['region_similarity'].fields.append(
  _REGIONSIMILARITYCALCULATOR.fields_by_name['thresholded_iou_similarity'])
_REGIONSIMILARITYCALCULATOR.fields_by_name['thresholded_iou_similarity'].containing_oneof = _REGIONSIMILARITYCALCULATOR.oneofs_by_name['region_similarity']
DESCRIPTOR.message_types_by_name['RegionSimilarityCalculator'] = _REGIONSIMILARITYCALCULATOR
DESCRIPTOR.message_types_by_name['NegSqDistSimilarity'] = _NEGSQDISTSIMILARITY
DESCRIPTOR.message_types_by_name['IouSimilarity'] = _IOUSIMILARITY
DESCRIPTOR.message_types_by_name['IoaSimilarity'] = _IOASIMILARITY
DESCRIPTOR.message_types_by_name['ThresholdedIouSimilarity'] = _THRESHOLDEDIOUSIMILARITY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RegionSimilarityCalculator = _reflection.GeneratedProtocolMessageType('RegionSimilarityCalculator', (_message.Message,), {
  'DESCRIPTOR' : _REGIONSIMILARITYCALCULATOR,
  '__module__' : 'nets.protos.region_similarity_calculator_pb2'
  # @@protoc_insertion_point(class_scope:nets.protos.RegionSimilarityCalculator)
  })
_sym_db.RegisterMessage(RegionSimilarityCalculator)

NegSqDistSimilarity = _reflection.GeneratedProtocolMessageType('NegSqDistSimilarity', (_message.Message,), {
  'DESCRIPTOR' : _NEGSQDISTSIMILARITY,
  '__module__' : 'nets.protos.region_similarity_calculator_pb2'
  # @@protoc_insertion_point(class_scope:nets.protos.NegSqDistSimilarity)
  })
_sym_db.RegisterMessage(NegSqDistSimilarity)

IouSimilarity = _reflection.GeneratedProtocolMessageType('IouSimilarity', (_message.Message,), {
  'DESCRIPTOR' : _IOUSIMILARITY,
  '__module__' : 'nets.protos.region_similarity_calculator_pb2'
  # @@protoc_insertion_point(class_scope:nets.protos.IouSimilarity)
  })
_sym_db.RegisterMessage(IouSimilarity)

IoaSimilarity = _reflection.GeneratedProtocolMessageType('IoaSimilarity', (_message.Message,), {
  'DESCRIPTOR' : _IOASIMILARITY,
  '__module__' : 'nets.protos.region_similarity_calculator_pb2'
  # @@protoc_insertion_point(class_scope:nets.protos.IoaSimilarity)
  })
_sym_db.RegisterMessage(IoaSimilarity)

ThresholdedIouSimilarity = _reflection.GeneratedProtocolMessageType('ThresholdedIouSimilarity', (_message.Message,), {
  'DESCRIPTOR' : _THRESHOLDEDIOUSIMILARITY,
  '__module__' : 'nets.protos.region_similarity_calculator_pb2'
  # @@protoc_insertion_point(class_scope:nets.protos.ThresholdedIouSimilarity)
  })
_sym_db.RegisterMessage(ThresholdedIouSimilarity)


# @@protoc_insertion_point(module_scope)
