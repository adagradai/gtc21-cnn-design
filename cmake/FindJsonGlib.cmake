# - Find json-glib
# This module looks for json-glib.
# This module defines the following values:
#    JSON_GLIB_FOUND
#    JSON_GLIB_INCLUDE_DIRS
#    JSON_GLIB_LIBRARIES

#=============================================================================
# Copyright Přemysl Janouch 2010
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
#=============================================================================

find_path (JSON_GLIB_INCLUDE_DIR json-glib/json-glib.h
        PATH_SUFFIXES json-glib-1.0)

find_path (JSON_GLIB_INCLUDE_DIR_GOBJECT glib-object.h
        PATH_SUFFIXES glib-2.0)


find_library (JSON_GLIB_LIBRARIES json-glib-1.0)

include (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS (JsonGlib DEFAULT_MSG JSON_GLIB_LIBRARIES
        JSON_GLIB_INCLUDE_DIR JSON_GLIB_INCLUDE_DIR_GOBJECT)

set (JSON_GLIB_INCLUDE_DIRS
        ${JSON_GLIB_INCLUDE_DIR}
        ${JSON_GLIB_INCLUDE_DIR_GOBJECT})

unset (JSON_GLIB_INCLUDE_DIR CACHE)
unset (JSON_GLIB_INCLUDE_DIR_GOBJECT CACHE)
mark_as_advanced (JSON_GLIB_LIBRARIES JSON_GLIB_INCLUDE_DIRS)
