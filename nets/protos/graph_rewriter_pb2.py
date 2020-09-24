# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nets/protos/graph_rewriter.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='nets/protos/graph_rewriter.proto',
  package='nets.protos',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=b'\n nets/protos/graph_rewriter.proto\x12\x0bnets.protos\"K\n\rGraphRewriter\x12/\n\x0cquantization\x18\x01 \x01(\x0b\x32\x19.nets.protos.Quantization*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02\"s\n\x0cQuantization\x12\x15\n\x05\x64\x65lay\x18\x01 \x01(\x05:\x06\x35\x30\x30\x30\x30\x30\x12\x16\n\x0bweight_bits\x18\x02 \x01(\x05:\x01\x38\x12\x1a\n\x0f\x61\x63tivation_bits\x18\x03 \x01(\x05:\x01\x38\x12\x18\n\tsymmetric\x18\x04 \x01(\x08:\x05\x66\x61lse'
)




_GRAPHREWRITER = _descriptor.Descriptor(
  name='GraphRewriter',
  full_name='nets.protos.GraphRewriter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='quantization', full_name='nets.protos.GraphRewriter.quantization', index=0,
      number=1, type=11, cpp_type=10, label=1,
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
  is_extendable=True,
  syntax='proto2',
  extension_ranges=[(1000, 536870912), ],
  oneofs=[
  ],
  serialized_start=49,
  serialized_end=124,
)


_QUANTIZATION = _descriptor.Descriptor(
  name='Quantization',
  full_name='nets.protos.Quantization',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='delay', full_name='nets.protos.Quantization.delay', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=500000,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weight_bits', full_name='nets.protos.Quantization.weight_bits', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=8,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='activation_bits', full_name='nets.protos.Quantization.activation_bits', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=8,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='symmetric', full_name='nets.protos.Quantization.symmetric', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
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
  serialized_start=126,
  serialized_end=241,
)

_GRAPHREWRITER.fields_by_name['quantization'].message_type = _QUANTIZATION
DESCRIPTOR.message_types_by_name['GraphRewriter'] = _GRAPHREWRITER
DESCRIPTOR.message_types_by_name['Quantization'] = _QUANTIZATION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GraphRewriter = _reflection.GeneratedProtocolMessageType('GraphRewriter', (_message.Message,), {
  'DESCRIPTOR' : _GRAPHREWRITER,
  '__module__' : 'nets.protos.graph_rewriter_pb2'
  # @@protoc_insertion_point(class_scope:nets.protos.GraphRewriter)
  })
_sym_db.RegisterMessage(GraphRewriter)

Quantization = _reflection.GeneratedProtocolMessageType('Quantization', (_message.Message,), {
  'DESCRIPTOR' : _QUANTIZATION,
  '__module__' : 'nets.protos.graph_rewriter_pb2'
  # @@protoc_insertion_point(class_scope:nets.protos.Quantization)
  })
_sym_db.RegisterMessage(Quantization)


# @@protoc_insertion_point(module_scope)
