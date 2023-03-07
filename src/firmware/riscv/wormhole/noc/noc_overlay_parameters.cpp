#include "noc_overlay_parameters.h"

using namespace Noc;

const std::vector<OverlayReg> OLP::registers = {
    {
        "STREAM_REMOTE_SRC",
        0,
        {
            {"STREAM_REMOTE_SRC_X", 0}
            ,{"STREAM_REMOTE_SRC_Y", 1}
            ,{"REMOTE_SRC_STREAM_ID", 2}
            ,{"STREAM_REMOTE_SRC_DEST_INDEX", 3}
            ,{"DRAM_READS__TRANS_SIZE_WORDS_LO", 4}
        },
        {
            {0, 0}
            ,{(STREAM_REMOTE_SRC_X+STREAM_REMOTE_SRC_X_WIDTH), 1}
            ,{(STREAM_REMOTE_SRC_Y+STREAM_REMOTE_SRC_Y_WIDTH), 2}
            ,{(REMOTE_SRC_STREAM_ID+REMOTE_SRC_STREAM_ID_WIDTH), 3}
        },
        {
            {
                "STREAM_REMOTE_SRC_X",
                0,
                NOC_ID_WIDTH,
                ""
            }
            ,{
                "STREAM_REMOTE_SRC_Y",
                (STREAM_REMOTE_SRC_X+STREAM_REMOTE_SRC_X_WIDTH),
                NOC_ID_WIDTH,
                ""
            }
            ,{
                "REMOTE_SRC_STREAM_ID",
                (STREAM_REMOTE_SRC_Y+STREAM_REMOTE_SRC_Y_WIDTH),
                STREAM_ID_WIDTH,
                ""
            }
            ,{
                "STREAM_REMOTE_SRC_DEST_INDEX",
                (REMOTE_SRC_STREAM_ID+REMOTE_SRC_STREAM_ID_WIDTH),
                STREAM_ID_WIDTH,
                ""
            }
            ,{
                "DRAM_READS__TRANS_SIZE_WORDS_LO",
                (STREAM_REMOTE_SRC_Y+STREAM_REMOTE_SRC_Y_WIDTH),
                12,
                ""
            }
        },
        "// Properties of the remote source stream (coorindates, stream ID, and this streams destination index).\n// Dont-care unless REMOTE_SOURCE == 1.\n"
    }
    ,{
        "STREAM_REMOTE_SRC_PHASE",
        1,
        {
            {"DRAM_READS__SCRATCH_1_PTR", 0}
            ,{"DRAM_READS__TRANS_SIZE_WORDS_HI", 1}
        },
        {
            {0, 0}
            ,{(DRAM_READS__SCRATCH_1_PTR+DRAM_READS__SCRATCH_1_PTR_WIDTH), 1}
        },
        {
            {
                "DRAM_READS__SCRATCH_1_PTR",
                0,
                19,
                ""
            }
            ,{
                "DRAM_READS__TRANS_SIZE_WORDS_HI",
                (DRAM_READS__SCRATCH_1_PTR+DRAM_READS__SCRATCH_1_PTR_WIDTH),
                1,
                ""
            }
        },
        "// Remote source phase (may be different from the destination stream phase.)\n// We use 20-bit phase ID, so phase count doesnt wrap until 1M phases. \n// Dont-care unless REMOTE_SOURCE == 1.\n"
    }
    ,{
        "STREAM_REMOTE_DEST",
        2,
        {
            {"STREAM_REMOTE_DEST_X", 0}
            ,{"STREAM_REMOTE_DEST_Y", 1}
            ,{"STREAM_REMOTE_DEST_STREAM_ID", 2}
        },
        {
            {0, 0}
            ,{(STREAM_REMOTE_DEST_X+STREAM_REMOTE_DEST_X_WIDTH), 1}
            ,{(STREAM_REMOTE_DEST_Y+STREAM_REMOTE_DEST_Y_WIDTH), 2}
        },
        {
            {
                "STREAM_REMOTE_DEST_X",
                0,
                NOC_ID_WIDTH,
                ""
            }
            ,{
                "STREAM_REMOTE_DEST_Y",
                (STREAM_REMOTE_DEST_X+STREAM_REMOTE_DEST_X_WIDTH),
                NOC_ID_WIDTH,
                ""
            }
            ,{
                "STREAM_REMOTE_DEST_STREAM_ID",
                (STREAM_REMOTE_DEST_Y+STREAM_REMOTE_DEST_Y_WIDTH),
                STREAM_ID_WIDTH,
                ""
            }
        },
        "// Properties of the remote destination stream (coorindates, stream ID).  Dont-care unless REMOTE_RECEIVER == 1.\n// If destination is multicast, this register specifies the starting coordinates of the destination\n// multicast group/rectangle. (The end coordinates are in STREAM_MCAST_DEST below.)\n"
    }
    ,{
        "STREAM_LOCAL_DEST",
        2,
        {
            {"STREAM_LOCAL_DEST_MSG_CLEAR_NUM", 0}
            ,{"STREAM_LOCAL_DEST_STREAM_ID", 1}
        },
        {
            {0, 0}
            ,{(STREAM_LOCAL_DEST_MSG_CLEAR_NUM+STREAM_LOCAL_DEST_MSG_CLEAR_NUM_WIDTH), 1}
        },
        {
            {
                "STREAM_LOCAL_DEST_MSG_CLEAR_NUM",
                0,
                12,
                ""
            }
            ,{
                "STREAM_LOCAL_DEST_STREAM_ID",
                (STREAM_LOCAL_DEST_MSG_CLEAR_NUM+STREAM_LOCAL_DEST_MSG_CLEAR_NUM_WIDTH),
                STREAM_ID_WIDTH,
                ""
            }
        },
        "// Properties of the local destination gather stream connection.\n// Dont-care unless LOCAL_RECEIVER == 1.\n// Shares register space with STREAM_REMOTE_DEST_REG_INDEX.\n"
    }
    ,{
        "STREAM_REMOTE_DEST_BUF_START",
        3,
        {
            {"DRAM_WRITES__SCRATCH_1_PTR_LO", 0}
        },
        {
            {0, 0}
        },
        {
            {
                "DRAM_WRITES__SCRATCH_1_PTR_LO",
                0,
                16,
                ""
            }
        },
        "// Start address (in words) of the remote destination stream memory buffer.\n"
    }
    ,{
        "STREAM_REMOTE_DEST_BUF_SIZE",
        4,
        {
            {"REMOTE_DEST_BUF_SIZE_WORDS", 0}
            ,{"DRAM_WRITES__SCRATCH_1_PTR_HI", 1}
        },
        {
            {0, 0}
        },
        {
            {
                "REMOTE_DEST_BUF_SIZE_WORDS",
                0,
                MEM_WORD_ADDR_WIDTH,
                ""
            }
            ,{
                "DRAM_WRITES__SCRATCH_1_PTR_HI",
                0,
                3,
                ""
            }
        },
        "// Size (in words) of the remote destination stream memory buffer.\n"
    }
    ,{
        "STREAM_REMOTE_DEST_WR_PTR",
        5,
        {
        },
        {
        },
        {
        },
        "// Write pointer for the remote destination stream memory buffer. \n// Can be written directly; automatically reset to 0 when \n// STREAM_REMOTE_DEST_BUF_START is written.\n"
    }
    ,{
        "STREAM_BUF_START",
        6,
        {
        },
        {
        },
        {
        },
        "// Start address (in words) of the memory buffer associated with this stream.\n"
    }
    ,{
        "STREAM_BUF_SIZE",
        7,
        {
        },
        {
        },
        {
        },
        "// Stream buffer size (in words).\n"
    }
    ,{
        "STREAM_MSG_INFO_PTR",
        8,
        {
        },
        {
        },
        {
        },
        "// Stream message info buffer address. \n//\n// This register needs to be initialized to the start of the message info buffer during \n// phase configuration.  Subsequently it will be incremented by hardware as data are read\n// from the buffer, thus doubling as the read pointer during phase execution. \n//\n// Stream hardware will assume that this buffer is large enough to hold info for all messages\n// within a phase, so unlike the buffer, it never needs to wrap.\n// \n// The buffer is filled automatically by snooping for streams with remote source. \n// For source enpoints, the buffer is written explicitly (along with the data buffer), after which \n// STREAM_NUM_MSGS_RECEIVED_INC is written to notify the stream that messages are available for\n// sending. \n// \n// Write pointer is also managed automatically by hardware, but can be read or reset using \n// STREAM_MSG_INFO_WR_PTR_REG. Write pointer is also reset when writing this register.\n"
    }
    ,{
        "STREAM_REMOTE_DEST_MSG_INFO_WR_PTR",
        9,
        {
        },
        {
        },
        {
        },
        "// Write pointer for the remote destination message info buffer. \n// Dont-care unless REMOTE_RECEIVER==1. \n// Needs to be initialized to the start of the message info buffer of the remote destination\n// at phase start, if destination is changed. \n// Subsequently its incremented automatically as messages are forwarded.\n"
    }
    ,{
        "STREAM_MISC_CFG",
        10,
        {
            {"INCOMING_DATA_NOC", 0}
            ,{"OUTGOING_DATA_NOC", 1}
            ,{"REMOTE_SRC_UPDATE_NOC", 2}
            ,{"LOCAL_SOURCES_CONNECTED", 3}
            ,{"SOURCE_ENDPOINT", 4}
            ,{"REMOTE_SOURCE", 5}
            ,{"RECEIVER_ENDPOINT", 6}
            ,{"LOCAL_RECEIVER", 7}
            ,{"REMOTE_RECEIVER", 8}
            ,{"PHASE_AUTO_CONFIG", 9}
            ,{"PHASE_AUTO_ADVANCE", 10}
            ,{"DATA_AUTO_SEND", 11}
            ,{"NEXT_PHASE_SRC_CHANGE", 12}
            ,{"NEXT_PHASE_DEST_CHANGE", 13}
            ,{"DATA_BUF_NO_FLOW_CTRL", 14}
            ,{"DEST_DATA_BUF_NO_FLOW_CTRL", 15}
            ,{"REMOTE_SRC_IS_MCAST", 16}
            ,{"NO_PREV_PHASE_OUTGOING_DATA_FLUSH", 17}
            ,{"UNICAST_VC_REG", 18}
            ,{"REG_UPDATE_VC_REG", 19}
        },
        {
            {0, 0}
            ,{(INCOMING_DATA_NOC+INCOMING_DATA_NOC_WIDTH), 1}
            ,{(OUTGOING_DATA_NOC+OUTGOING_DATA_NOC_WIDTH), 2}
            ,{(REMOTE_SRC_UPDATE_NOC+REMOTE_SRC_UPDATE_NOC_WIDTH), 3}
            ,{(LOCAL_SOURCES_CONNECTED+LOCAL_SOURCES_CONNECTED_WIDTH), 4}
            ,{(SOURCE_ENDPOINT+SOURCE_ENDPOINT_WIDTH), 5}
            ,{(REMOTE_SOURCE+REMOTE_SOURCE_WIDTH), 6}
            ,{(RECEIVER_ENDPOINT+RECEIVER_ENDPOINT_WIDTH), 7}
            ,{(LOCAL_RECEIVER+LOCAL_RECEIVER_WIDTH), 8}
            ,{(REMOTE_RECEIVER+REMOTE_RECEIVER_WIDTH), 9}
            ,{(PHASE_AUTO_CONFIG+PHASE_AUTO_CONFIG_WIDTH), 10}
            ,{(PHASE_AUTO_ADVANCE+PHASE_AUTO_ADVANCE_WIDTH), 11}
            ,{(DATA_AUTO_SEND+DATA_AUTO_SEND_WIDTH), 12}
            ,{(NEXT_PHASE_SRC_CHANGE+NEXT_PHASE_SRC_CHANGE_WIDTH), 13}
            ,{(NEXT_PHASE_DEST_CHANGE+NEXT_PHASE_DEST_CHANGE_WIDTH), 14}
            ,{(DATA_BUF_NO_FLOW_CTRL+DATA_BUF_NO_FLOW_CTRL_WIDTH), 15}
            ,{(DEST_DATA_BUF_NO_FLOW_CTRL+DEST_DATA_BUF_NO_FLOW_CTRL_WIDTH), 16}
            ,{(REMOTE_SRC_IS_MCAST+REMOTE_SRC_IS_MCAST_WIDTH), 17}
            ,{(NO_PREV_PHASE_OUTGOING_DATA_FLUSH+NO_PREV_PHASE_OUTGOING_DATA_FLUSH_WIDTH), 18}
            ,{(UNICAST_VC_REG+UNICAST_VC_REG_WIDTH), 19}
        },
        {
            {
                "INCOMING_DATA_NOC",
                0,
                NOC_NUM_WIDTH,
                ""
            }
            ,{
                "OUTGOING_DATA_NOC",
                (INCOMING_DATA_NOC+INCOMING_DATA_NOC_WIDTH),
                NOC_NUM_WIDTH,
                ""
            }
            ,{
                "REMOTE_SRC_UPDATE_NOC",
                (OUTGOING_DATA_NOC+OUTGOING_DATA_NOC_WIDTH),
                NOC_NUM_WIDTH,
                ""
            }
            ,{
                "LOCAL_SOURCES_CONNECTED",
                (REMOTE_SRC_UPDATE_NOC+REMOTE_SRC_UPDATE_NOC_WIDTH),
                1,
                ""
            }
            ,{
                "SOURCE_ENDPOINT",
                (LOCAL_SOURCES_CONNECTED+LOCAL_SOURCES_CONNECTED_WIDTH),
                1,
                ""
            }
            ,{
                "REMOTE_SOURCE",
                (SOURCE_ENDPOINT+SOURCE_ENDPOINT_WIDTH),
                1,
                ""
            }
            ,{
                "RECEIVER_ENDPOINT",
                (REMOTE_SOURCE+REMOTE_SOURCE_WIDTH),
                1,
                ""
            }
            ,{
                "LOCAL_RECEIVER",
                (RECEIVER_ENDPOINT+RECEIVER_ENDPOINT_WIDTH),
                1,
                ""
            }
            ,{
                "REMOTE_RECEIVER",
                (LOCAL_RECEIVER+LOCAL_RECEIVER_WIDTH),
                1,
                ""
            }
            ,{
                "PHASE_AUTO_CONFIG",
                (REMOTE_RECEIVER+REMOTE_RECEIVER_WIDTH),
                1,
                ""
            }
            ,{
                "PHASE_AUTO_ADVANCE",
                (PHASE_AUTO_CONFIG+PHASE_AUTO_CONFIG_WIDTH),
                1,
                ""
            }
            ,{
                "DATA_AUTO_SEND",
                (PHASE_AUTO_ADVANCE+PHASE_AUTO_ADVANCE_WIDTH),
                1,
                ""
            }
            ,{
                "NEXT_PHASE_SRC_CHANGE",
                (DATA_AUTO_SEND+DATA_AUTO_SEND_WIDTH),
                1,
                ""
            }
            ,{
                "NEXT_PHASE_DEST_CHANGE",
                (NEXT_PHASE_SRC_CHANGE+NEXT_PHASE_SRC_CHANGE_WIDTH),
                1,
                ""
            }
            ,{
                "DATA_BUF_NO_FLOW_CTRL",
                (NEXT_PHASE_DEST_CHANGE+NEXT_PHASE_DEST_CHANGE_WIDTH),
                1,
                "// set if REMOTE_SOURCE==1 and the buffer is large enough to accept full phase data without wrapping:\n"
            }
            ,{
                "DEST_DATA_BUF_NO_FLOW_CTRL",
                (DATA_BUF_NO_FLOW_CTRL+DATA_BUF_NO_FLOW_CTRL_WIDTH),
                1,
                "// set if REMOTE_RECEIVER==1 and the destination buffer is large enough to accept full phase data without wrapping:\n"
            }
            ,{
                "REMOTE_SRC_IS_MCAST",
                (DEST_DATA_BUF_NO_FLOW_CTRL+DEST_DATA_BUF_NO_FLOW_CTRL_WIDTH),
                1,
                "// set if REMOTE_SOURCE==1 and has mulicast enabled (i.e. this stream is part of a multicast group)\n"
            }
            ,{
                "NO_PREV_PHASE_OUTGOING_DATA_FLUSH",
                (REMOTE_SRC_IS_MCAST+REMOTE_SRC_IS_MCAST_WIDTH),
                1,
                "// set if no need to flush outgoing remote data from previous phase\n"
            }
            ,{
                "UNICAST_VC_REG",
                (NO_PREV_PHASE_OUTGOING_DATA_FLUSH+NO_PREV_PHASE_OUTGOING_DATA_FLUSH_WIDTH),
                3,
                "// set to one of the values (0-5) to select which VC unicast requests will be sent on\n"
            }
            ,{
                "REG_UPDATE_VC_REG",
                (UNICAST_VC_REG+UNICAST_VC_REG_WIDTH),
                3,
                "// set to one of the values (0-5) to select which VC control flow updates will be sent on\n"
            }
        },
        "// The ID of NOCs used for incoming and outgoing data, followed by misc. stream configuration options:\n//   * Source - set exactly one of these to 1:\n//        SOURCE_ENDPOINT = source is local math/packer\n//        REMOTE_SOURCE = source is remote sender stream\n//        LOCAL_SOURCES_CONNECTED = source is one or more local connected streams\n//   * Destination - set one or zero of these to 1:\n//        RECEIVER_ENDPOINT = stream is read by local unpacker/math\n//        REMOTE_RECEIVER = stream forwards data to a remote destination or multicast group\n//        LOCAL_RECEIVER = stream is connected to a local destination stream\n//        None set = stream just stores data in a local buffer, without forwarding/clearing, and \n//                   finishes the phase once all messages have been received\n//   * Phase/data forward options:\n//      PHASE_AUTO_CONFIG = set to 1 for stream to fetch next phase configuration automatically.\n//      PHASE_AUTO_ADVANCE = set to 1 for stream to advance to next phase automatically \n//            (otherwise need to write STREAM_PHASE_ADVANCE below)\n//      DATA_AUTO_SEND = set to 1 to forward data automatically based on read/write pointers; \n//             set to 0 to forward data only when STREAM_NEXT_MSG_SEND is written\n"
    }
    ,{
        "STREAM_CURR_PHASE",
        11,
        {
        },
        {
        },
        {
        },
        "// Current phase number executed by the stream.\n"
    }
    ,{
        "STREAM_PHASE_AUTO_CFG_PTR",
        12,
        {
        },
        {
        },
        {
        },
        "// Pointer to the stream auto-config data. Initialized to the start of\n// the auto-config structure at workload start, automatically updated\n// subsequenty. \n// Specified as byte address, needs to be multiple of 4B.\n"
    }
    ,{
        "STREAM_MCAST_DEST",
        13,
        {
            {"STREAM_MCAST_END_X", 0}
            ,{"STREAM_MCAST_END_Y", 1}
            ,{"STREAM_MCAST_EN", 2}
            ,{"STREAM_MCAST_LINKED", 3}
            ,{"STREAM_MCAST_VC", 4}
            ,{"STREAM_MCAST_NO_PATH_RES", 5}
            ,{"STREAM_MCAST_XY", 6}
        },
        {
            {0, 0}
            ,{(STREAM_MCAST_END_X+STREAM_MCAST_END_X_WIDTH), 1}
            ,{(STREAM_MCAST_END_Y+STREAM_MCAST_END_Y_WIDTH), 2}
            ,{(STREAM_MCAST_EN+STREAM_MCAST_EN_WIDTH), 3}
            ,{(STREAM_MCAST_LINKED+STREAM_MCAST_LINKED_WIDTH), 4}
            ,{(STREAM_MCAST_VC+STREAM_MCAST_VC_WIDTH), 5}
            ,{(STREAM_MCAST_NO_PATH_RES+STREAM_MCAST_NO_PATH_RES_WIDTH), 6}
        },
        {
            {
                "STREAM_MCAST_END_X",
                0,
                NOC_ID_WIDTH,
                ""
            }
            ,{
                "STREAM_MCAST_END_Y",
                (STREAM_MCAST_END_X+STREAM_MCAST_END_X_WIDTH),
                NOC_ID_WIDTH,
                ""
            }
            ,{
                "STREAM_MCAST_EN",
                (STREAM_MCAST_END_Y+STREAM_MCAST_END_Y_WIDTH),
                1,
                ""
            }
            ,{
                "STREAM_MCAST_LINKED",
                (STREAM_MCAST_EN+STREAM_MCAST_EN_WIDTH),
                1,
                ""
            }
            ,{
                "STREAM_MCAST_VC",
                (STREAM_MCAST_LINKED+STREAM_MCAST_LINKED_WIDTH),
                1,
                "// Set to 0 to select VC 4, and 1 to select VC 5 (default 0)\n"
            }
            ,{
                "STREAM_MCAST_NO_PATH_RES",
                (STREAM_MCAST_VC+STREAM_MCAST_VC_WIDTH),
                1,
                ""
            }
            ,{
                "STREAM_MCAST_XY",
                (STREAM_MCAST_NO_PATH_RES+STREAM_MCAST_NO_PATH_RES_WIDTH),
                1,
                ""
            }
        },
        "// Destination spec for multicasting streams. STREAM_MCAST_END_X/Y are\n// the end coordinate for the multicast rectangle, with the ones from \n// STREAM_REMOTE_DEST taken as start. \n// Dont-care if STREAM_MCAST_EN == 0.\n"
    }
    ,{
        "STREAM_MCAST_DEST_NUM",
        14,
        {
        },
        {
        },
        {
        },
        "// Number of multicast destinations (dont-care for non-multicast streams)\n"
    }
    ,{
        "STREAM_GATHER",
        15,
        {
            {"MSG_ARB_GROUP_SIZE", 0}
            ,{"MSG_SRC_IN_ORDER_FWD", 1}
        },
        {
            {0, 0}
            ,{(MSG_ARB_GROUP_SIZE+MSG_ARB_GROUP_SIZE_WIDTH), 1}
        },
        {
            {
                "MSG_ARB_GROUP_SIZE",
                0,
                3,
                ""
            }
            ,{
                "MSG_SRC_IN_ORDER_FWD",
                (MSG_ARB_GROUP_SIZE+MSG_ARB_GROUP_SIZE_WIDTH),
                1,
                ""
            }
        },
        "// Specifies MSG_ARB_GROUP_SIZE. Valid values are 1 (round-robin\n// arbitration between each incoming stream) or 4 (round-robin arbitration\n// between groups of 4 incoming streams).\n"
    }
    ,{
        "STREAM_MSG_SRC_IN_ORDER_FWD_NUM_MSGS",
        16,
        {
        },
        {
        },
        {
        },
        "// When using in-order message forwarding, number of messages after which the source\n// pointer goes back to zero (without phase change).\n// Dont-care if STREAM_MCAST_EN == 0 or MSG_SRC_IN_ORDER_FWD == 0.\n"
    }
    ,{
        "STREAM_MSG_HEADER_FORMAT",
        17,
        {
            {"MSG_HEADER_WORD_CNT_OFFSET", 0}
            ,{"MSG_HEADER_WORD_CNT_BITS", 1}
        },
        {
            {0, 0}
            ,{(MSG_HEADER_WORD_CNT_OFFSET+MSG_HEADER_WORD_CNT_OFFSET_WIDTH), 1}
        },
        {
            {
                "MSG_HEADER_WORD_CNT_OFFSET",
                0,
                MEM_WORD_BIT_OFFSET_WIDTH,
                ""
            }
            ,{
                "MSG_HEADER_WORD_CNT_BITS",
                (MSG_HEADER_WORD_CNT_OFFSET+MSG_HEADER_WORD_CNT_OFFSET_WIDTH),
                MEM_WORD_BIT_OFFSET_WIDTH,
                ""
            }
        },
        "// Offset & size of the size field in the message header. Only valid offsets are multiples of 8\n// (i.e. byte-aligned).\n"
    }
    ,{
        "STREAM_NUM_MSGS_RECEIVED",
        18,
        {
        },
        {
        },
        {
        },
        "// Number of received & stored messages (read-only). \n// To get the total number of messages penidng in memory read \n// STREAM_NUM_MSGS_RECEIVED_IN_BUF_AND_MEM_REG_INDEX\n"
    }
    ,{
        "STREAM_NEXT_RECEIVED_MSG_ADDR",
        19,
        {
        },
        {
        },
        {
        },
        "// Memory address (in words) of the next in line received message (read-only).\n"
    }
    ,{
        "STREAM_NEXT_RECEIVED_MSG_SIZE",
        20,
        {
        },
        {
        },
        {
        },
        "// Size in words of the next in line received message (read-only).\n"
    }
    ,{
        "STREAM_MSG_INFO_CLEAR",
        21,
        {
        },
        {
        },
        {
        },
        "// Clear message info for one or more stored messages.  Only valid values are 1, 2, or 4. \n// No effect on the read pointer. \n// Should be used only for streams where RECEIVER_ENDPOINT == 1.\n"
    }
    ,{
        "STREAM_MSG_DATA_CLEAR",
        22,
        {
        },
        {
        },
        {
        },
        "// Move read pointer & reclaim buffer space for one or more stored messages.  \n// Sends flow control update to the source if REMOTE_SOURCE==1. \n// Only valid values are 1, 2, or 4. \n// Should be used only for streams where RECEIVER_ENDPOINT == 1, after \n// STREAM_MSG_INFO_CLEAR_REG has been written with the same value.\n"
    }
    ,{
        "STREAM_NEXT_MSG_SEND",
        23,
        {
        },
        {
        },
        {
        },
        "// Write to send the next in line stored message. Used when DATA_AUTO_SEND == 0.\n"
    }
    ,{
        "STREAM_RD_PTR",
        24,
        {
        },
        {
        },
        {
        },
        "// Read pointer value (word offset relative to buffer start). Can be updated by\n// writing the register (e.g. to force resend). \n// Value does not guarantee that all data up to the current value have been sent\n// off (forwarding command may be  ongoing).  To find out free space in the buffer,\n// read STREAM_BUF_SPACE_AVAILABLE. \n// Automatically reset to 0 when STREAM_BUF_START_REG is updated.\n"
    }
    ,{
        "STREAM_WR_PTR",
        25,
        {
        },
        {
        },
        {
        },
        "// Write pointer value (word offset relative to buffer start). \n// Can be read to determine the location at which to write new data. \n// In normal operation, should be updated only by writing \n// STREAM_NUM_MSGS_RECEIVED_INC_REG or STREAM_SOURCE_ENDPOINT_NEW_MSG_INFO_REG.\n"
    }
    ,{
        "STREAM_MSG_INFO_WR_PTR",
        26,
        {
        },
        {
        },
        {
        },
        "// Write pointer value for message info buffer (absolute word address). \n// In normal operation, should be updated only by writing \n// STREAM_NUM_MSGS_RECEIVED_INC_REG or STREAM_SOURCE_ENDPOINT_NEW_MSG_INFO_REG.\n"
    }
    ,{
        "STREAM_PHASE_ADVANCE",
        27,
        {
        },
        {
        },
        {
        },
        "// Write-only. Write 1 to advance to the next phase if PHASE_AUTO_ADVANCE == 0.\n"
    }
    ,{
        "STREAM_BUF_SPACE_AVAILABLE",
        28,
        {
        },
        {
        },
        {
        },
        "// Available buffer space at the stream (in 16B words). \n// Source cant send data unless available space > 0.\n"
    }
    ,{
        "STREAM_SOURCE_ENDPOINT_NEW_MSG_INFO",
        29,
        {
            {"SOURCE_ENDPOINT_NEW_MSG_ADDR", 0}
            ,{"SOURCE_ENDPOINT_NEW_MSG_SIZE", 1}
        },
        {
            {0, 0}
            ,{(SOURCE_ENDPOINT_NEW_MSG_ADDR+SOURCE_ENDPOINT_NEW_MSG_ADDR_WIDTH), 1}
        },
        {
            {
                "SOURCE_ENDPOINT_NEW_MSG_ADDR",
                0,
                MEM_WORD_ADDR_WIDTH,
                ""
            }
            ,{
                "SOURCE_ENDPOINT_NEW_MSG_SIZE",
                (SOURCE_ENDPOINT_NEW_MSG_ADDR+SOURCE_ENDPOINT_NEW_MSG_ADDR_WIDTH),
                (32-MEM_WORD_ADDR_WIDTH),
                ""
            }
        },
        "// For endpoints with SOURCE_ENDPOINT == 1, this register is for firmware \n// to register new message for sending. \n// This updates the msg_info register structure directly, rather than writing to the message info\n// buffer in memory.\n// Must not be written when the message info register structure is full, or if\n// there are message info entries in the memory buffer. (This would cause a race\n// condition.)\n"
    }
    ,{
        "STREAM_NUM_MSGS_RECEIVED_INC",
        30,
        {
            {"SOURCE_ENDPOINT_NEW_MSGS_NUM", 0}
            ,{"SOURCE_ENDPOINT_NEW_MSGS_TOTAL_SIZE", 1}
        },
        {
            {0, 0}
            ,{(SOURCE_ENDPOINT_NEW_MSGS_NUM+SOURCE_ENDPOINT_NEW_MSGS_NUM_WIDTH), 1}
        },
        {
            {
                "SOURCE_ENDPOINT_NEW_MSGS_NUM",
                0,
                12,
                ""
            }
            ,{
                "SOURCE_ENDPOINT_NEW_MSGS_TOTAL_SIZE",
                (SOURCE_ENDPOINT_NEW_MSGS_NUM+SOURCE_ENDPOINT_NEW_MSGS_NUM_WIDTH),
                MEM_WORD_ADDR_WIDTH,
                ""
            }
        },
        "// For endpoints with SOURCE_ENDPOINT == 1, this register is for firmware \n// to update the number of messages whose data & header are available in the memory buffer.\n// Hardware register is incremented atomically if sending of previous messages is in progress.\n"
    }
    ,{
        "STREAM_RESET",
        31,
        {
        },
        {
        },
        {
        },
        "// Write to reset & stop stream.\n"
    }
    ,{
        "STREAM_DEST_PHASE_READY_UPDATE",
        32,
        {
            {"PHASE_READY_DEST_NUM", 0}
            ,{"PHASE_READY_NUM", 1}
            ,{"PHASE_READY_MCAST", 2}
            ,{"PHASE_READY_TWO_WAY_RESP", 3}
        },
        {
            {0, 0}
            ,{(PHASE_READY_DEST_NUM+PHASE_READY_DEST_NUM_WIDTH), 1}
            ,{(PHASE_READY_NUM+PHASE_READY_NUM_WIDTH), 2}
            ,{(PHASE_READY_MCAST+PHASE_READY_MCAST_WIDTH), 3}
        },
        {
            {
                "PHASE_READY_DEST_NUM",
                0,
                6,
                ""
            }
            ,{
                "PHASE_READY_NUM",
                (PHASE_READY_DEST_NUM+PHASE_READY_DEST_NUM_WIDTH),
                20,
                ""
            }
            ,{
                "PHASE_READY_MCAST",
                (PHASE_READY_NUM+PHASE_READY_NUM_WIDTH),
                1,
                "// set if this stream is part of multicast group (i.e. if REMOTE_SRC_IS_MCAST==1)\n"
            }
            ,{
                "PHASE_READY_TWO_WAY_RESP",
                (PHASE_READY_MCAST+PHASE_READY_MCAST_WIDTH),
                1,
                "// set if the message is in response to 2-way handshake\n"
            }
        },
        "// Write phase number to indicate destination ready for the given phase. \n// (This is done automatically by stream hardware when starting a phase with REMOTE_SOURCE=1.)\n// The phase number is the one indicated by STREAM_REMOTE_SRC_PHASE_REG at destination. \n// This register is mapped to the shared destination ready table, not a per-stream register.\n// (Stream index is taken from the register address, and stored into the table along with the\n// phase number.)\n"
    }
    ,{
        "STREAM_SRC_READY_UPDATE",
        33,
        {
            {"STREAM_REMOTE_RDY_SRC_X", 0}
            ,{"STREAM_REMOTE_RDY_SRC_Y", 1}
            ,{"REMOTE_RDY_SRC_STREAM_ID", 2}
        },
        {
            {0, 0}
            ,{(STREAM_REMOTE_RDY_SRC_X+STREAM_REMOTE_RDY_SRC_X_WIDTH), 1}
            ,{(STREAM_REMOTE_RDY_SRC_Y+STREAM_REMOTE_RDY_SRC_Y_WIDTH), 2}
        },
        {
            {
                "STREAM_REMOTE_RDY_SRC_X",
                0,
                NOC_ID_WIDTH,
                ""
            }
            ,{
                "STREAM_REMOTE_RDY_SRC_Y",
                (STREAM_REMOTE_RDY_SRC_X+STREAM_REMOTE_RDY_SRC_X_WIDTH),
                NOC_ID_WIDTH,
                ""
            }
            ,{
                "REMOTE_RDY_SRC_STREAM_ID",
                (STREAM_REMOTE_RDY_SRC_Y+STREAM_REMOTE_RDY_SRC_Y_WIDTH),
                STREAM_ID_WIDTH,
                ""
            }
        },
        "// Source ready message register for two-way handshake (sent by source in \n// case destination ready entry is not found in the table). \n// If received by a stream that already sent its ready update, it prompts resending.\n"
    }
    ,{
        "STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE",
        34,
        {
            {"REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM", 0}
            ,{"REMOTE_DEST_BUF_WORDS_FREE_INC", 1}
        },
        {
            {0, 0}
            ,{(REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM+REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM_WIDTH), 1}
        },
        {
            {
                "REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM",
                0,
                6,
                ""
            }
            ,{
                "REMOTE_DEST_BUF_WORDS_FREE_INC",
                (REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM+REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM_WIDTH),
                MEM_WORD_ADDR_WIDTH,
                ""
            }
        },
        "// Update available buffer space at remote destination stream. \n// this is rd_ptr increment issued when a message is forwarded\n"
    }
    ,{
        "STREAM_WAIT_STATUS",
        35,
        {
            {"WAIT_SW_PHASE_ADVANCE_SIGNAL", 0}
            ,{"WAIT_PREV_PHASE_DATA_FLUSH", 1}
            ,{"MSG_FWD_ONGOING", 2}
            ,{"STREAM_CURR_STATE", 3}
        },
        {
            {0, 0}
            ,{(WAIT_SW_PHASE_ADVANCE_SIGNAL+WAIT_SW_PHASE_ADVANCE_SIGNAL_WIDTH), 1}
            ,{(WAIT_PREV_PHASE_DATA_FLUSH+WAIT_PREV_PHASE_DATA_FLUSH_WIDTH), 2}
            ,{(MSG_FWD_ONGOING+MSG_FWD_ONGOING_WIDTH), 3}
        },
        {
            {
                "WAIT_SW_PHASE_ADVANCE_SIGNAL",
                0,
                1,
                "// Set when stream is in START state with auto-config disabled, or if auto-config is enabled\n// but PHASE_AUTO_ADVANCE=0\n"
            }
            ,{
                "WAIT_PREV_PHASE_DATA_FLUSH",
                (WAIT_SW_PHASE_ADVANCE_SIGNAL+WAIT_SW_PHASE_ADVANCE_SIGNAL_WIDTH),
                1,
                "// Set when stream has configured the current phase, but waits data from the previous one to be flushed.\n"
            }
            ,{
                "MSG_FWD_ONGOING",
                (WAIT_PREV_PHASE_DATA_FLUSH+WAIT_PREV_PHASE_DATA_FLUSH_WIDTH),
                1,
                "// Set when stream is in data forwarding state.\n"
            }
            ,{
                "STREAM_CURR_STATE",
                (MSG_FWD_ONGOING+MSG_FWD_ONGOING_WIDTH),
                4,
                ""
            }
        },
        "// Status info for the stream.\n"
    }
    ,{
        "STREAM_PHASE_AUTO_CFG_HEADER",
        36,
        {
            {"PHASE_NUM_INCR", 0}
            ,{"CURR_PHASE_NUM_MSGS", 1}
            ,{"NEXT_PHASE_NUM_CFG_REG_WRITES", 2}
        },
        {
            {0, 0}
            ,{(PHASE_NUM_INCR+PHASE_NUM_INCR_WIDTH), 1}
            ,{(CURR_PHASE_NUM_MSGS+CURR_PHASE_NUM_MSGS_WIDTH), 2}
        },
        {
            {
                "PHASE_NUM_INCR",
                0,
                12,
                ""
            }
            ,{
                "CURR_PHASE_NUM_MSGS",
                (PHASE_NUM_INCR+PHASE_NUM_INCR_WIDTH),
                12,
                ""
            }
            ,{
                "NEXT_PHASE_NUM_CFG_REG_WRITES",
                (CURR_PHASE_NUM_MSGS+CURR_PHASE_NUM_MSGS_WIDTH),
                8,
                ""
            }
        },
        "// Register corresponding to the auto-configuration header. Written by each auto-config access\n// at phase start, can be also written by software for initial configuration or if auto-config\n// is disabled. \n// PHASE_NUM_INCR is phase number increment relative to the previous executed phase (or 0 right\n// after reset). The increment happens after auto-config is done, and before the phase is executed.\n// (Therefore reading  STREAM_CURR_PHASE_REG while auto-config is ongoing, or if it hasnt started\n// yet, may return the old phase number.)\n// This enables up to 2^12-1 phases to be skipped. If more phases need to be skipped, it is\n// necessary to insert an intermediate phase with zero messages, whose only purpose is to provide\n// an additional skip offset.\n"
    }
    ,{
        "STREAM_PERF_CONFIG",
        37,
        {
            {"CLOCK_GATING_EN", 0}
            ,{"CLOCK_GATING_HYST", 1}
            ,{"PARTIAL_SEND_WORDS_THR", 2}
        },
        {
            {0, 0}
            ,{(CLOCK_GATING_EN+CLOCK_GATING_EN_WIDTH), 1}
            ,{(CLOCK_GATING_HYST+CLOCK_GATING_HYST_WIDTH), 2}
        },
        {
            {
                "CLOCK_GATING_EN",
                0,
                1,
                ""
            }
            ,{
                "CLOCK_GATING_HYST",
                (CLOCK_GATING_EN+CLOCK_GATING_EN_WIDTH),
                7,
                ""
            }
            ,{
                "PARTIAL_SEND_WORDS_THR",
                (CLOCK_GATING_HYST+CLOCK_GATING_HYST_WIDTH),
                8,
                "// PARTIAL_SEND_WORDS_THR contols the minimum number of 16-byte words of a tile to accumulate in a relay stream before sending it off to the destination.\n// If the size of the tile is less than or equal to PARTIAL_SEND_WORDS_THR, then this feild is ignored.\n// Default is 16 words\n"
            }
        },
        "// Should be written only for stream 0, applies to all streams.\n"
    }
    ,{
        "STREAM_MSG_GROUP_ZERO_MASK_AND",
        38,
        {
        },
        {
        },
        {
        },
        "// AND value of zero masks for the pending message group. \n// (Header bits [95:64].)\n// Read-only.  Valid only for receiver endpoint streams.\n"
    }
    ,{
        "STREAM_MSG_INFO_FULL",
        39,
        {
        },
        {
        },
        {
        },
        "// Returns 1 if the message info register is full (read-only).\n"
    }
    ,{
        "STREAM_MEM_BUF_SPACE_AVAILABLE_ACK_THRESHOLD",
        40,
        {
        },
        {
        },
        {
        },
        "// 4-bit wide register that determines the threshold at which a stream\n// with remote source sends an update message to STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE.\n// Dont-care unless REMOTE_SOURCE==1.  \n// Values:\n//   value[3:0] == 0 => disable threshold. Acks send as soon as any data are cleared/forwarded. \n//   value[3:0] >  0 => threshold calculated according to the following formula:\n//         if (value[3])\n//              threshold = buf_size - (buf_size >> value[2:0])\n//         else \n//              threshold = (buf_size >> value[2:0])\n//\n// This enables setting thresholds of buf_size/2, buf_size/4, buf_size/8, ... buf_size/256, \n// as well as  3*buf_size/4, 7*buf_size/8, etc.\n"
    }
    ,{
        "STREAM_MSG_INFO_CAN_PUSH_NEW_MSG",
        41,
        {
        },
        {
        },
        {
        },
        "// Returns 1 if the message info register can accept new message push (read-only). \n// Equivalent to checking the condition:\n//   (STREAM_MSG_INFO_FULL_REG_INDEX == 0) && (STREAM_MSG_INFO_PTR_REG_INDEX == STREAM_MSG_INFO_WR_PTR_REG_INDEX)\n// (I.e. ther is free space in the msg info register, and we dont have any message info headers in the\n//  memory buffer about to be fetched.)\n"
    }
    ,{
        "STREAM_MSG_GROUP_COMPRESS",
        42,
        {
        },
        {
        },
        {
        },
        "// Concat compress flags from 4 tiles in the pending message group.\n// (Header bit 52.)\n// Read-only.  Valid only for receiver endpoint streams.\n"
    }
    ,{
        "STREAM_GATHER_CLEAR",
        43,
        {
            {"MSG_LOCAL_STREAM_CLEAR_NUM", 0}
            ,{"MSG_GROUP_STREAM_CLEAR_TYPE", 1}
        },
        {
            {0, 0}
            ,{(MSG_LOCAL_STREAM_CLEAR_NUM+MSG_LOCAL_STREAM_CLEAR_NUM_WIDTH), 1}
        },
        {
            {
                "MSG_LOCAL_STREAM_CLEAR_NUM",
                0,
                16,
                ""
            }
            ,{
                "MSG_GROUP_STREAM_CLEAR_TYPE",
                (MSG_LOCAL_STREAM_CLEAR_NUM+MSG_LOCAL_STREAM_CLEAR_NUM_WIDTH),
                1,
                ""
            }
        },
        "// Msg_LOCAL_STREAM_CLEAR_NUM specifies the number of messages that should \n// be cleared from a gather stream before moving onto the next stream. \n// When MSG_ARB_GROUP_SIZE > 1, the order of clearing the streams can be selected\n// with MSG_GROUP_STREAM_CLEAR_TYPE. 0 = clear the whole group MSG_LOCAL_STREAM_CLEAR_NUM times,\n// 1 = clear each stream of the group MSG_LOCAL_STREAM_CLEAR_NUM times before\n// moving onto the next stream in the group.\n"
    }
    ,{
        "STREAM_REMOTE_DEST_TRAFFIC_PRIORITY",
        44,
        {
        },
        {
        },
        {
        },
        "// Priority for traffic sent to remote destination. \n// Valid only for streams capable of remote sending. \n// 4-bit value. \n// Set to 0 to send traffic under round-robin arbitration. \n// Set to 1-15 for priority arbitration (higher values are higher priority).\n"
    }
    ,{
        "STREAM_DEBUG_STATUS_SEL",
        45,
        {
            {"DEBUG_STATUS_STREAM_ID_SEL", 0}
            ,{"DISABLE_DEST_READY_TABLE", 1}
        },
        {
            {0, 0}
            ,{(DEBUG_STATUS_STREAM_ID_SEL+DEBUG_STATUS_STREAM_ID_SEL_WIDTH), 1}
        },
        {
            {
                "DEBUG_STATUS_STREAM_ID_SEL",
                0,
                STREAM_ID_WIDTH,
                ""
            }
            ,{
                "DISABLE_DEST_READY_TABLE",
                (DEBUG_STATUS_STREAM_ID_SEL+DEBUG_STATUS_STREAM_ID_SEL_WIDTH),
                1,
                ""
            }
        },
        "// Debug bus stream selection. Write the stream id for the stream that you want exposed on the debug bus\n// This register only exists in stream 0.\n"
    }
    ,{
        "STREAM_DEBUG_ASSERTIONS",
        46,
        {
        },
        {
        },
        {
        },
        "// Debugging: Non-zero value indicates an invalid stream operation occured.\n// Sticky, write 1 to clear.\n"
    }
    ,{
        "STREAM_NUM_MSGS_RECEIVED_IN_BUF_AND_MEM",
        47,
        {
        },
        {
        },
        {
        },
        "// Only in receiver endpoint streams (stream 4 and 5)\n// Read-only. Tells you the number of tiles that have arrived in L1\n"
    }
    ,{
        "STREAM_LOCAL_SRC_MASK",
        48,
        {
        },
        {
        },
        {
        },
        "// Bit mask of connnected local source. Dont care if LOCAL_SOURCES_CONNECTED == 0.\n// Mask segments [23:0], [47:24], and [63:48] are at indexes STREAM_LOCAL_SRC_MASK_REG_INDEX, \n// STREAM_LOCAL_SRC_MASK_REG_INDEX+1, STREAM_LOCAL_SRC_MASK_REG_INDEX+2.\n"
    }
    ,{
        "STREAM_RECEIVER_ENDPOINT_SET_MSG_HEADER",
        60,
        {
        },
        {
        },
        {
        },
        "// For receiver endpoint streams that expose the full message header bus to unpacker,\n// write this register to specify the full header in case the stream is not snooping\n// a remote source but instead also works as a source endpoint. \n// Write (STREAM_RECEIVER_ENDPOINT_SET_MSG_HEADER_REG_INDEX+i) to set bits [i*32 +: 32]\n// of the message header for the next message, prior to writing STREAM_SOURCE_ENDPOINT_NEW_MSG_INFO_REG_INDEX.\n"
    }
    ,{
        "STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE",
        64,
        {
            {"REMOTE_DEST_WORDS_FREE", 0}
        },
        {
            {0, 0}
        },
        {
            {
                "REMOTE_DEST_WORDS_FREE",
                0,
                MEM_WORD_ADDR_WIDTH,
                ""
            }
        },
        "// Available buffer space at remote destination stream(s). \n// Dont care unless REMOTE_RECEIVER == 1. \n// Source cant send data unless WORDS_FREE > 0.  \n// Read-only; updated automatically to maximum value when \n// STREAM_REMOTE_DEST_BUF_SIZE_REG is updated. \n// For multicast streams, values for successive destinations are at \n// subsequent indexes (STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX+1, \n// STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX+2, etc.).\n"
    }
    ,{
        "STREAM_RECEIVER_MSG_INFO",
        128,
        {
        },
        {
        },
        {
        },
        "// Read-only register view of the bits on the o_full_msg_info bus. \n// Exposed as 32-bit read-only registers starting at this index.\n"
    }
    ,{
        "STREAM_DEBUG_STATUS",
        224,
        {
        },
        {
        },
        {
        },
        "// Read-only register that exposes internal states of the stream.\n// Useful for debugging. Valid 32-bit data from STREAM_DEBUG_STATUS_REG_INDEX + 0 to STREAM_DEBUG_STATUS_REG_INDEX + 9\n"
    }
    ,{
        "STREAM_BLOB_AUTO_CFG_DONE",
        234,
        {
        },
        {
        },
        {
        },
        "// 32 bit register. Each bit denotes whether the corresponding stream has completed its blob run and is in idle state.\n// Resets to 0 upon starting a new stream run. Initially all are 0 to exclude streams that might not be used.\n// Can be manually reset to 0 by writing 1 to the corresponding bit.\n// Exists only in stream 0\n"
    }
    ,{
        "STREAM_REMOTE_DEST_BUF_START_HI",
        242,
        {
        },
        {
        },
        {
        },
        "// High bits for STREAM_REMOTE_DEST_BUF_START\n"
    }
    ,{
        "STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_HI",
        243,
        {
        },
        {
        },
        {
        },
        "// High bits for STREAM_REMOTE_DEST_MSG_INFO_WR_PTR\n"
    }
    ,{
        "STREAM_CURR_PHASE_BASE",
        244,
        {
        },
        {
        },
        {
        },
        "// Actual phase number executed is STREAM_CURR_PHASE_BASE_REG_INDEX + STREAM_CURR_PHASE_REG_INDEX\n// When reprogramming this register you must also reprogram STREAM_CURR_PHASE_REG_INDEX and STREAM_REMOTE_SRC_PHASE_REG_INDEX\n"
    }
    ,{
        "STREAM_PHASE_AUTO_CFG_PTR_BASE",
        245,
        {
        },
        {
        },
        {
        },
        "// Actual address accessed will be STREAM_PHASE_AUTO_CFG_PTR_BASE_REG_INDEX + STREAM_PHASE_AUTO_CFG_PTR_REG_INDEX\n// When reprogramming this register you must also reprogram STREAM_PHASE_AUTO_CFG_PTR_REG_INDEX\n"
    }
    ,{
        "STREAM_BLOB_NEXT_AUTO_CFG_DONE",
        246,
        {
            {"BLOB_NEXT_AUTO_CFG_DONE_STREAM_ID", 0}
            ,{"BLOB_NEXT_AUTO_CFG_DONE_VALID", 1}
        },
        {
            {0, 0}
            ,{16, 1}
        },
        {
            {
                "BLOB_NEXT_AUTO_CFG_DONE_STREAM_ID",
                0,
                STREAM_ID_WIDTH,
                ""
            }
            ,{
                "BLOB_NEXT_AUTO_CFG_DONE_VALID",
                16,
                1,
                ""
            }
        },
        "// Reading this register will give you a stream id of a stream that finished its blob (according to STREAM_BLOB_AUTO_CFG_DONE_REG_INDEX)\n// Subsequent reads will give you the next stream, untill all streams are read, after which it will loop\n// This register is only valid if BLOB_NEXT_AUTO_CFG_DONE_VALID is set (i.e. if STREAM_BLOB_AUTO_CFG_DONE_REG_INDEX non-zero)\n// Exists only in stream 0\n"
    }
    ,{
        "FIRMWARE_SCRATCH",
        247,
        {
        },
        {
        },
        {
        },
        "// Scratch location for firmware usage\n// Guarantees that no side-effects occur in Overlay hardware\n"
    }
    ,{
        "STREAM_SCRATCH",
        248,
        {
        },
        {
        },
        {
        },
        "// Scratch registers\n// Exists only in streams 0-3 and 8-11\n// Data can be stored at [23:0] from STREAM_SCRATCH_REG_INDEX + 0 to STREAM_SCRATCH_REG_INDEX + 5\n// Can be loaded through overlay blobs.\n"
    }
    ,{
        "STREAM_SCRATCH_0",
        248,
        {
            {"NCRISC_TRANS_EN", 0}
            ,{"NCRISC_TRANS_EN_IRQ_ON_BLOB_END", 1}
            ,{"NCRISC_CMD_ID", 2}
            ,{"NEXT_NRISC_PIC_INT_ON_PHASE", 3}
        },
        {
            {0, 0}
            ,{(NCRISC_TRANS_EN + NCRISC_TRANS_EN_WIDTH), 1}
            ,{(NCRISC_TRANS_EN_IRQ_ON_BLOB_END + NCRISC_TRANS_EN_IRQ_ON_BLOB_END_WIDTH), 2}
            ,{(NCRISC_CMD_ID + NCRISC_CMD_ID_WIDTH), 3}
        },
        {
            {
                "NCRISC_TRANS_EN",
                0,
                1,
                ""
            }
            ,{
                "NCRISC_TRANS_EN_IRQ_ON_BLOB_END",
                (NCRISC_TRANS_EN + NCRISC_TRANS_EN_WIDTH),
                1,
                ""
            }
            ,{
                "NCRISC_CMD_ID",
                (NCRISC_TRANS_EN_IRQ_ON_BLOB_END + NCRISC_TRANS_EN_IRQ_ON_BLOB_END_WIDTH),
                3,
                ""
            }
            ,{
                "NEXT_NRISC_PIC_INT_ON_PHASE",
                (NCRISC_CMD_ID + NCRISC_CMD_ID_WIDTH),
                19,
                "// Kept for compatibility with grayskull, but doesnt not exist anymore in wormhole\n"
            }
        },
        ""
    }
    ,{
        "STREAM_SCRATCH_1",
        249,
        {
            {"DRAM_FIFO_RD_PTR_WORDS_LO", 0}
            ,{"NCRISC_LOOP_COUNT", 1}
            ,{"NCRISC_INIT_ENABLE_BLOB_DONE_IRQ", 2}
            ,{"NCRISC_INIT_DISABLE_BLOB_DONE_IRQ", 3}
        },
        {
            {0, 0}
            ,{(NCRISC_INIT_ENABLE_BLOB_DONE_IRQ + NCRISC_INIT_ENABLE_BLOB_DONE_IRQ_WIDTH), 3}
        },
        {
            {
                "DRAM_FIFO_RD_PTR_WORDS_LO",
                0,
                24,
                ""
            }
            ,{
                "NCRISC_LOOP_COUNT",
                0,
                24,
                ""
            }
            ,{
                "NCRISC_INIT_ENABLE_BLOB_DONE_IRQ",
                0,
                1,
                ""
            }
            ,{
                "NCRISC_INIT_DISABLE_BLOB_DONE_IRQ",
                (NCRISC_INIT_ENABLE_BLOB_DONE_IRQ + NCRISC_INIT_ENABLE_BLOB_DONE_IRQ_WIDTH),
                1,
                ""
            }
        },
        ""
    }
    ,{
        "STREAM_SCRATCH_2",
        250,
        {
            {"DRAM_FIFO_RD_PTR_WORDS_HI", 0}
            ,{"DRAM_FIFO_WR_PTR_WORDS_LO", 1}
            ,{"NCRISC_TOTAL_LOOP_ITER", 2}
        },
        {
            {0, 0}
            ,{(DRAM_FIFO_RD_PTR_WORDS_HI + DRAM_FIFO_RD_PTR_WORDS_HI_WIDTH), 1}
        },
        {
            {
                "DRAM_FIFO_RD_PTR_WORDS_HI",
                0,
                4,
                ""
            }
            ,{
                "DRAM_FIFO_WR_PTR_WORDS_LO",
                (DRAM_FIFO_RD_PTR_WORDS_HI + DRAM_FIFO_RD_PTR_WORDS_HI_WIDTH),
                20,
                ""
            }
            ,{
                "NCRISC_TOTAL_LOOP_ITER",
                0,
                24,
                ""
            }
        },
        ""
    }
    ,{
        "STREAM_SCRATCH_3",
        251,
        {
            {"DRAM_FIFO_WR_PTR_WORDS_HI", 0}
            ,{"DRAM_FIFO_CAPACITY_PTR_WORDS_LO", 1}
            ,{"NCRISC_LOOP_INCR", 2}
            ,{"NCRISC_LOOP_BACK_NUM_CFG_REG_WRITES", 3}
        },
        {
            {0, 0}
            ,{(DRAM_FIFO_WR_PTR_WORDS_HI + DRAM_FIFO_WR_PTR_WORDS_HI_WIDTH), 1}
            ,{(NCRISC_LOOP_INCR+NCRISC_LOOP_INCR_WIDTH), 3}
        },
        {
            {
                "DRAM_FIFO_WR_PTR_WORDS_HI",
                0,
                8,
                ""
            }
            ,{
                "DRAM_FIFO_CAPACITY_PTR_WORDS_LO",
                (DRAM_FIFO_WR_PTR_WORDS_HI + DRAM_FIFO_WR_PTR_WORDS_HI_WIDTH),
                16,
                ""
            }
            ,{
                "NCRISC_LOOP_INCR",
                0,
                16,
                ""
            }
            ,{
                "NCRISC_LOOP_BACK_NUM_CFG_REG_WRITES",
                (NCRISC_LOOP_INCR+NCRISC_LOOP_INCR_WIDTH),
                8,
                ""
            }
        },
        ""
    }
    ,{
        "STREAM_SCRATCH_4",
        252,
        {
            {"DRAM_FIFO_CAPACITY_PTR_WORDS_HI", 0}
            ,{"DRAM_FIFO_BASE_ADDR_WORDS_LO", 1}
            ,{"NCRISC_LOOP_BACK_AUTO_CFG_PTR", 2}
        },
        {
            {0, 0}
            ,{(DRAM_FIFO_CAPACITY_PTR_WORDS_HI + DRAM_FIFO_CAPACITY_PTR_WORDS_HI_WIDTH), 1}
        },
        {
            {
                "DRAM_FIFO_CAPACITY_PTR_WORDS_HI",
                0,
                12,
                ""
            }
            ,{
                "DRAM_FIFO_BASE_ADDR_WORDS_LO",
                (DRAM_FIFO_CAPACITY_PTR_WORDS_HI + DRAM_FIFO_CAPACITY_PTR_WORDS_HI_WIDTH),
                12,
                ""
            }
            ,{
                "NCRISC_LOOP_BACK_AUTO_CFG_PTR",
                0,
                24,
                ""
            }
        },
        ""
    }
    ,{
        "STREAM_SCRATCH_5",
        253,
        {
            {"DRAM_FIFO_BASE_ADDR_WORDS_HI", 0}
            ,{"DRAM_EN_BLOCKING", 1}
            ,{"DRAM_DATA_STRUCTURE_IS_LUT", 2}
            ,{"DRAM_RESET_RD_PTR_TO_BASE_ON_EMPTY", 3}
            ,{"DRAM_RESET_WR_PTR_TO_BASE_ON_FULL", 4}
            ,{"DRAM_NO_PTR_UPDATE_ON_PHASE_END", 5}
            ,{"DRAM_WR_BUFFER_FLUSH_AND_RST_PTRS", 6}
            ,{"NCRISC_LOOP_NEXT_PIC_INT_ON_PHASE", 7}
        },
        {
            {0, 0}
            ,{(DRAM_FIFO_BASE_ADDR_WORDS_HI + DRAM_FIFO_BASE_ADDR_WORDS_HI_WIDTH), 1}
            ,{(DRAM_EN_BLOCKING + DRAM_EN_BLOCKING_WIDTH), 2}
            ,{(DRAM_DATA_STRUCTURE_IS_LUT + DRAM_DATA_STRUCTURE_IS_LUT_WIDTH), 3}
            ,{(DRAM_RESET_RD_PTR_TO_BASE_ON_EMPTY + DRAM_RESET_RD_PTR_TO_BASE_ON_EMPTY_WIDTH), 4}
            ,{(DRAM_RESET_WR_PTR_TO_BASE_ON_FULL + DRAM_RESET_WR_PTR_TO_BASE_ON_FULL_WIDTH), 5}
            ,{(DRAM_NO_PTR_UPDATE_ON_PHASE_END + DRAM_NO_PTR_UPDATE_ON_PHASE_END_WIDTH), 6}
        },
        {
            {
                "DRAM_FIFO_BASE_ADDR_WORDS_HI",
                0,
                16,
                ""
            }
            ,{
                "DRAM_EN_BLOCKING",
                (DRAM_FIFO_BASE_ADDR_WORDS_HI + DRAM_FIFO_BASE_ADDR_WORDS_HI_WIDTH),
                1,
                "// Processes the read or write operation to completeion without processing other dram streams in the meantime\n"
            }
            ,{
                "DRAM_DATA_STRUCTURE_IS_LUT",
                (DRAM_EN_BLOCKING + DRAM_EN_BLOCKING_WIDTH),
                1,
                "// Fifo structure in dram holds a dram pointer and size that is used as indirection to a tile in dram\n"
            }
            ,{
                "DRAM_RESET_RD_PTR_TO_BASE_ON_EMPTY",
                (DRAM_DATA_STRUCTURE_IS_LUT + DRAM_DATA_STRUCTURE_IS_LUT_WIDTH),
                1,
                "// During a dram read, if its detected that the fifo is empty the ncrisc will reset the read pointer back to base\n// Its expected that there is no host interaction\n"
            }
            ,{
                "DRAM_RESET_WR_PTR_TO_BASE_ON_FULL",
                (DRAM_RESET_RD_PTR_TO_BASE_ON_EMPTY + DRAM_RESET_RD_PTR_TO_BASE_ON_EMPTY_WIDTH),
                1,
                "// During a dram write, if its detected that the fifo is full the ncrisc will reset the write pointer back to base. Old data will be overwritten.\n// Its expected that there is no host interaction\n"
            }
            ,{
                "DRAM_NO_PTR_UPDATE_ON_PHASE_END",
                (DRAM_RESET_WR_PTR_TO_BASE_ON_FULL + DRAM_RESET_WR_PTR_TO_BASE_ON_FULL_WIDTH),
                1,
                "// The internal ncrisc rd/wr pointers will not be updated at phase end\n// Its expected that there is no host interaction\n"
            }
            ,{
                "DRAM_WR_BUFFER_FLUSH_AND_RST_PTRS",
                (DRAM_NO_PTR_UPDATE_ON_PHASE_END + DRAM_NO_PTR_UPDATE_ON_PHASE_END_WIDTH),
                1,
                "// Before ending the phase the ncrisc will wait until the host has emptied the write buffer and then reset the read and write pointers to base\n// This can be used for hosts that do not want to track wrapping\n// The host must be aware of this behaviour for this functionality to work\n"
            }
            ,{
                "NCRISC_LOOP_NEXT_PIC_INT_ON_PHASE",
                0,
                20,
                ""
            }
        },
        ""
    }
};

const std::unordered_map<std::string, std::uint32_t> OLP::registers_by_name = {
    {"STREAM_REMOTE_SRC", 0}
    ,{"STREAM_REMOTE_SRC_PHASE", 1}
    ,{"STREAM_REMOTE_DEST", 2}
    ,{"STREAM_LOCAL_DEST", 3}
    ,{"STREAM_REMOTE_DEST_BUF_START", 4}
    ,{"STREAM_REMOTE_DEST_BUF_SIZE", 5}
    ,{"STREAM_REMOTE_DEST_WR_PTR", 6}
    ,{"STREAM_BUF_START", 7}
    ,{"STREAM_BUF_SIZE", 8}
    ,{"STREAM_MSG_INFO_PTR", 9}
    ,{"STREAM_REMOTE_DEST_MSG_INFO_WR_PTR", 10}
    ,{"STREAM_MISC_CFG", 11}
    ,{"STREAM_CURR_PHASE", 12}
    ,{"STREAM_PHASE_AUTO_CFG_PTR", 13}
    ,{"STREAM_MCAST_DEST", 14}
    ,{"STREAM_MCAST_DEST_NUM", 15}
    ,{"STREAM_GATHER", 16}
    ,{"STREAM_MSG_SRC_IN_ORDER_FWD_NUM_MSGS", 17}
    ,{"STREAM_MSG_HEADER_FORMAT", 18}
    ,{"STREAM_NUM_MSGS_RECEIVED", 19}
    ,{"STREAM_NEXT_RECEIVED_MSG_ADDR", 20}
    ,{"STREAM_NEXT_RECEIVED_MSG_SIZE", 21}
    ,{"STREAM_MSG_INFO_CLEAR", 22}
    ,{"STREAM_MSG_DATA_CLEAR", 23}
    ,{"STREAM_NEXT_MSG_SEND", 24}
    ,{"STREAM_RD_PTR", 25}
    ,{"STREAM_WR_PTR", 26}
    ,{"STREAM_MSG_INFO_WR_PTR", 27}
    ,{"STREAM_PHASE_ADVANCE", 28}
    ,{"STREAM_BUF_SPACE_AVAILABLE", 29}
    ,{"STREAM_SOURCE_ENDPOINT_NEW_MSG_INFO", 30}
    ,{"STREAM_NUM_MSGS_RECEIVED_INC", 31}
    ,{"STREAM_RESET", 32}
    ,{"STREAM_DEST_PHASE_READY_UPDATE", 33}
    ,{"STREAM_SRC_READY_UPDATE", 34}
    ,{"STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE", 35}
    ,{"STREAM_WAIT_STATUS", 36}
    ,{"STREAM_PHASE_AUTO_CFG_HEADER", 37}
    ,{"STREAM_PERF_CONFIG", 38}
    ,{"STREAM_MSG_GROUP_ZERO_MASK_AND", 39}
    ,{"STREAM_MSG_INFO_FULL", 40}
    ,{"STREAM_MEM_BUF_SPACE_AVAILABLE_ACK_THRESHOLD", 41}
    ,{"STREAM_MSG_INFO_CAN_PUSH_NEW_MSG", 42}
    ,{"STREAM_MSG_GROUP_COMPRESS", 43}
    ,{"STREAM_GATHER_CLEAR", 44}
    ,{"STREAM_REMOTE_DEST_TRAFFIC_PRIORITY", 45}
    ,{"STREAM_DEBUG_STATUS_SEL", 46}
    ,{"STREAM_DEBUG_ASSERTIONS", 47}
    ,{"STREAM_NUM_MSGS_RECEIVED_IN_BUF_AND_MEM", 48}
    ,{"STREAM_LOCAL_SRC_MASK", 49}
    ,{"STREAM_RECEIVER_ENDPOINT_SET_MSG_HEADER", 50}
    ,{"STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE", 51}
    ,{"STREAM_RECEIVER_MSG_INFO", 52}
    ,{"STREAM_DEBUG_STATUS", 53}
    ,{"STREAM_BLOB_AUTO_CFG_DONE", 54}
    ,{"STREAM_REMOTE_DEST_BUF_START_HI", 55}
    ,{"STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_HI", 56}
    ,{"STREAM_CURR_PHASE_BASE", 57}
    ,{"STREAM_PHASE_AUTO_CFG_PTR_BASE", 58}
    ,{"STREAM_BLOB_NEXT_AUTO_CFG_DONE", 59}
    ,{"FIRMWARE_SCRATCH", 60}
    ,{"STREAM_SCRATCH", 61}
    ,{"STREAM_SCRATCH_0", 62}
    ,{"STREAM_SCRATCH_1", 63}
    ,{"STREAM_SCRATCH_2", 64}
    ,{"STREAM_SCRATCH_3", 65}
    ,{"STREAM_SCRATCH_4", 66}
    ,{"STREAM_SCRATCH_5", 67}
};

const std::unordered_map<std::uint32_t, std::uint32_t> OLP::registers_by_index = {
    {0, 0}
    ,{1, 1}
    ,{2, 2}
    ,{3, 4}
    ,{4, 5}
    ,{5, 6}
    ,{6, 7}
    ,{7, 8}
    ,{8, 9}
    ,{9, 10}
    ,{10, 11}
    ,{11, 12}
    ,{12, 13}
    ,{13, 14}
    ,{14, 15}
    ,{15, 16}
    ,{16, 17}
    ,{17, 18}
    ,{18, 19}
    ,{19, 20}
    ,{20, 21}
    ,{21, 22}
    ,{22, 23}
    ,{23, 24}
    ,{24, 25}
    ,{25, 26}
    ,{26, 27}
    ,{27, 28}
    ,{28, 29}
    ,{29, 30}
    ,{30, 31}
    ,{31, 32}
    ,{32, 33}
    ,{33, 34}
    ,{34, 35}
    ,{35, 36}
    ,{36, 37}
    ,{37, 38}
    ,{38, 39}
    ,{39, 40}
    ,{40, 41}
    ,{41, 42}
    ,{42, 43}
    ,{43, 44}
    ,{44, 45}
    ,{45, 46}
    ,{46, 47}
    ,{47, 48}
    ,{48, 49}
    ,{60, 50}
    ,{64, 51}
    ,{128, 52}
    ,{224, 53}
    ,{234, 54}
    ,{242, 55}
    ,{243, 56}
    ,{244, 57}
    ,{245, 58}
    ,{246, 59}
    ,{247, 60}
    ,{248, 61}
    ,{249, 63}
    ,{250, 64}
    ,{251, 65}
    ,{252, 66}
    ,{253, 67}
};

const std::vector<OverlayField> OLP::fields = {
    {
        "STREAM_REMOTE_SRC_X",
        0,
        NOC_ID_WIDTH,
        ""
    }
    ,{
        "STREAM_REMOTE_SRC_Y",
        (STREAM_REMOTE_SRC_X+STREAM_REMOTE_SRC_X_WIDTH),
        NOC_ID_WIDTH,
        ""
    }
    ,{
        "REMOTE_SRC_STREAM_ID",
        (STREAM_REMOTE_SRC_Y+STREAM_REMOTE_SRC_Y_WIDTH),
        STREAM_ID_WIDTH,
        ""
    }
    ,{
        "STREAM_REMOTE_SRC_DEST_INDEX",
        (REMOTE_SRC_STREAM_ID+REMOTE_SRC_STREAM_ID_WIDTH),
        STREAM_ID_WIDTH,
        ""
    }
    ,{
        "DRAM_READS__TRANS_SIZE_WORDS_LO",
        (STREAM_REMOTE_SRC_Y+STREAM_REMOTE_SRC_Y_WIDTH),
        12,
        ""
    }
    ,{
        "DRAM_READS__SCRATCH_1_PTR",
        0,
        19,
        ""
    }
    ,{
        "DRAM_READS__TRANS_SIZE_WORDS_HI",
        (DRAM_READS__SCRATCH_1_PTR+DRAM_READS__SCRATCH_1_PTR_WIDTH),
        1,
        ""
    }
    ,{
        "STREAM_REMOTE_DEST_X",
        0,
        NOC_ID_WIDTH,
        ""
    }
    ,{
        "STREAM_REMOTE_DEST_Y",
        (STREAM_REMOTE_DEST_X+STREAM_REMOTE_DEST_X_WIDTH),
        NOC_ID_WIDTH,
        ""
    }
    ,{
        "STREAM_REMOTE_DEST_STREAM_ID",
        (STREAM_REMOTE_DEST_Y+STREAM_REMOTE_DEST_Y_WIDTH),
        STREAM_ID_WIDTH,
        ""
    }
    ,{
        "STREAM_LOCAL_DEST_MSG_CLEAR_NUM",
        0,
        12,
        ""
    }
    ,{
        "STREAM_LOCAL_DEST_STREAM_ID",
        (STREAM_LOCAL_DEST_MSG_CLEAR_NUM+STREAM_LOCAL_DEST_MSG_CLEAR_NUM_WIDTH),
        STREAM_ID_WIDTH,
        ""
    }
    ,{
        "DRAM_WRITES__SCRATCH_1_PTR_LO",
        0,
        16,
        ""
    }
    ,{
        "REMOTE_DEST_BUF_SIZE_WORDS",
        0,
        MEM_WORD_ADDR_WIDTH,
        ""
    }
    ,{
        "DRAM_WRITES__SCRATCH_1_PTR_HI",
        0,
        3,
        ""
    }
    ,{
        "INCOMING_DATA_NOC",
        0,
        NOC_NUM_WIDTH,
        ""
    }
    ,{
        "OUTGOING_DATA_NOC",
        (INCOMING_DATA_NOC+INCOMING_DATA_NOC_WIDTH),
        NOC_NUM_WIDTH,
        ""
    }
    ,{
        "REMOTE_SRC_UPDATE_NOC",
        (OUTGOING_DATA_NOC+OUTGOING_DATA_NOC_WIDTH),
        NOC_NUM_WIDTH,
        ""
    }
    ,{
        "LOCAL_SOURCES_CONNECTED",
        (REMOTE_SRC_UPDATE_NOC+REMOTE_SRC_UPDATE_NOC_WIDTH),
        1,
        ""
    }
    ,{
        "SOURCE_ENDPOINT",
        (LOCAL_SOURCES_CONNECTED+LOCAL_SOURCES_CONNECTED_WIDTH),
        1,
        ""
    }
    ,{
        "REMOTE_SOURCE",
        (SOURCE_ENDPOINT+SOURCE_ENDPOINT_WIDTH),
        1,
        ""
    }
    ,{
        "RECEIVER_ENDPOINT",
        (REMOTE_SOURCE+REMOTE_SOURCE_WIDTH),
        1,
        ""
    }
    ,{
        "LOCAL_RECEIVER",
        (RECEIVER_ENDPOINT+RECEIVER_ENDPOINT_WIDTH),
        1,
        ""
    }
    ,{
        "REMOTE_RECEIVER",
        (LOCAL_RECEIVER+LOCAL_RECEIVER_WIDTH),
        1,
        ""
    }
    ,{
        "PHASE_AUTO_CONFIG",
        (REMOTE_RECEIVER+REMOTE_RECEIVER_WIDTH),
        1,
        ""
    }
    ,{
        "PHASE_AUTO_ADVANCE",
        (PHASE_AUTO_CONFIG+PHASE_AUTO_CONFIG_WIDTH),
        1,
        ""
    }
    ,{
        "DATA_AUTO_SEND",
        (PHASE_AUTO_ADVANCE+PHASE_AUTO_ADVANCE_WIDTH),
        1,
        ""
    }
    ,{
        "NEXT_PHASE_SRC_CHANGE",
        (DATA_AUTO_SEND+DATA_AUTO_SEND_WIDTH),
        1,
        ""
    }
    ,{
        "NEXT_PHASE_DEST_CHANGE",
        (NEXT_PHASE_SRC_CHANGE+NEXT_PHASE_SRC_CHANGE_WIDTH),
        1,
        ""
    }
    ,{
        "DATA_BUF_NO_FLOW_CTRL",
        (NEXT_PHASE_DEST_CHANGE+NEXT_PHASE_DEST_CHANGE_WIDTH),
        1,
        "// set if REMOTE_SOURCE==1 and the buffer is large enough to accept full phase data without wrapping:\n"
    }
    ,{
        "DEST_DATA_BUF_NO_FLOW_CTRL",
        (DATA_BUF_NO_FLOW_CTRL+DATA_BUF_NO_FLOW_CTRL_WIDTH),
        1,
        "// set if REMOTE_RECEIVER==1 and the destination buffer is large enough to accept full phase data without wrapping:\n"
    }
    ,{
        "REMOTE_SRC_IS_MCAST",
        (DEST_DATA_BUF_NO_FLOW_CTRL+DEST_DATA_BUF_NO_FLOW_CTRL_WIDTH),
        1,
        "// set if REMOTE_SOURCE==1 and has mulicast enabled (i.e. this stream is part of a multicast group)\n"
    }
    ,{
        "NO_PREV_PHASE_OUTGOING_DATA_FLUSH",
        (REMOTE_SRC_IS_MCAST+REMOTE_SRC_IS_MCAST_WIDTH),
        1,
        "// set if no need to flush outgoing remote data from previous phase\n"
    }
    ,{
        "UNICAST_VC_REG",
        (NO_PREV_PHASE_OUTGOING_DATA_FLUSH+NO_PREV_PHASE_OUTGOING_DATA_FLUSH_WIDTH),
        3,
        "// set to one of the values (0-5) to select which VC unicast requests will be sent on\n"
    }
    ,{
        "REG_UPDATE_VC_REG",
        (UNICAST_VC_REG+UNICAST_VC_REG_WIDTH),
        3,
        "// set to one of the values (0-5) to select which VC control flow updates will be sent on\n"
    }
    ,{
        "STREAM_MCAST_END_X",
        0,
        NOC_ID_WIDTH,
        ""
    }
    ,{
        "STREAM_MCAST_END_Y",
        (STREAM_MCAST_END_X+STREAM_MCAST_END_X_WIDTH),
        NOC_ID_WIDTH,
        ""
    }
    ,{
        "STREAM_MCAST_EN",
        (STREAM_MCAST_END_Y+STREAM_MCAST_END_Y_WIDTH),
        1,
        ""
    }
    ,{
        "STREAM_MCAST_LINKED",
        (STREAM_MCAST_EN+STREAM_MCAST_EN_WIDTH),
        1,
        ""
    }
    ,{
        "STREAM_MCAST_VC",
        (STREAM_MCAST_LINKED+STREAM_MCAST_LINKED_WIDTH),
        1,
        "// Set to 0 to select VC 4, and 1 to select VC 5 (default 0)\n"
    }
    ,{
        "STREAM_MCAST_NO_PATH_RES",
        (STREAM_MCAST_VC+STREAM_MCAST_VC_WIDTH),
        1,
        ""
    }
    ,{
        "STREAM_MCAST_XY",
        (STREAM_MCAST_NO_PATH_RES+STREAM_MCAST_NO_PATH_RES_WIDTH),
        1,
        ""
    }
    ,{
        "MSG_ARB_GROUP_SIZE",
        0,
        3,
        ""
    }
    ,{
        "MSG_SRC_IN_ORDER_FWD",
        (MSG_ARB_GROUP_SIZE+MSG_ARB_GROUP_SIZE_WIDTH),
        1,
        ""
    }
    ,{
        "MSG_HEADER_WORD_CNT_OFFSET",
        0,
        MEM_WORD_BIT_OFFSET_WIDTH,
        ""
    }
    ,{
        "MSG_HEADER_WORD_CNT_BITS",
        (MSG_HEADER_WORD_CNT_OFFSET+MSG_HEADER_WORD_CNT_OFFSET_WIDTH),
        MEM_WORD_BIT_OFFSET_WIDTH,
        ""
    }
    ,{
        "SOURCE_ENDPOINT_NEW_MSG_ADDR",
        0,
        MEM_WORD_ADDR_WIDTH,
        ""
    }
    ,{
        "SOURCE_ENDPOINT_NEW_MSG_SIZE",
        (SOURCE_ENDPOINT_NEW_MSG_ADDR+SOURCE_ENDPOINT_NEW_MSG_ADDR_WIDTH),
        (32-MEM_WORD_ADDR_WIDTH),
        ""
    }
    ,{
        "SOURCE_ENDPOINT_NEW_MSGS_NUM",
        0,
        12,
        ""
    }
    ,{
        "SOURCE_ENDPOINT_NEW_MSGS_TOTAL_SIZE",
        (SOURCE_ENDPOINT_NEW_MSGS_NUM+SOURCE_ENDPOINT_NEW_MSGS_NUM_WIDTH),
        MEM_WORD_ADDR_WIDTH,
        ""
    }
    ,{
        "PHASE_READY_DEST_NUM",
        0,
        6,
        ""
    }
    ,{
        "PHASE_READY_NUM",
        (PHASE_READY_DEST_NUM+PHASE_READY_DEST_NUM_WIDTH),
        20,
        ""
    }
    ,{
        "PHASE_READY_MCAST",
        (PHASE_READY_NUM+PHASE_READY_NUM_WIDTH),
        1,
        "// set if this stream is part of multicast group (i.e. if REMOTE_SRC_IS_MCAST==1)\n"
    }
    ,{
        "PHASE_READY_TWO_WAY_RESP",
        (PHASE_READY_MCAST+PHASE_READY_MCAST_WIDTH),
        1,
        "// set if the message is in response to 2-way handshake\n"
    }
    ,{
        "STREAM_REMOTE_RDY_SRC_X",
        0,
        NOC_ID_WIDTH,
        ""
    }
    ,{
        "STREAM_REMOTE_RDY_SRC_Y",
        (STREAM_REMOTE_RDY_SRC_X+STREAM_REMOTE_RDY_SRC_X_WIDTH),
        NOC_ID_WIDTH,
        ""
    }
    ,{
        "REMOTE_RDY_SRC_STREAM_ID",
        (STREAM_REMOTE_RDY_SRC_Y+STREAM_REMOTE_RDY_SRC_Y_WIDTH),
        STREAM_ID_WIDTH,
        ""
    }
    ,{
        "REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM",
        0,
        6,
        ""
    }
    ,{
        "REMOTE_DEST_BUF_WORDS_FREE_INC",
        (REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM+REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM_WIDTH),
        MEM_WORD_ADDR_WIDTH,
        ""
    }
    ,{
        "WAIT_SW_PHASE_ADVANCE_SIGNAL",
        0,
        1,
        "// Set when stream is in START state with auto-config disabled, or if auto-config is enabled\n// but PHASE_AUTO_ADVANCE=0\n"
    }
    ,{
        "WAIT_PREV_PHASE_DATA_FLUSH",
        (WAIT_SW_PHASE_ADVANCE_SIGNAL+WAIT_SW_PHASE_ADVANCE_SIGNAL_WIDTH),
        1,
        "// Set when stream has configured the current phase, but waits data from the previous one to be flushed.\n"
    }
    ,{
        "MSG_FWD_ONGOING",
        (WAIT_PREV_PHASE_DATA_FLUSH+WAIT_PREV_PHASE_DATA_FLUSH_WIDTH),
        1,
        "// Set when stream is in data forwarding state.\n"
    }
    ,{
        "STREAM_CURR_STATE",
        (MSG_FWD_ONGOING+MSG_FWD_ONGOING_WIDTH),
        4,
        ""
    }
    ,{
        "PHASE_NUM_INCR",
        0,
        12,
        ""
    }
    ,{
        "CURR_PHASE_NUM_MSGS",
        (PHASE_NUM_INCR+PHASE_NUM_INCR_WIDTH),
        12,
        ""
    }
    ,{
        "NEXT_PHASE_NUM_CFG_REG_WRITES",
        (CURR_PHASE_NUM_MSGS+CURR_PHASE_NUM_MSGS_WIDTH),
        8,
        ""
    }
    ,{
        "CLOCK_GATING_EN",
        0,
        1,
        ""
    }
    ,{
        "CLOCK_GATING_HYST",
        (CLOCK_GATING_EN+CLOCK_GATING_EN_WIDTH),
        7,
        ""
    }
    ,{
        "PARTIAL_SEND_WORDS_THR",
        (CLOCK_GATING_HYST+CLOCK_GATING_HYST_WIDTH),
        8,
        "// PARTIAL_SEND_WORDS_THR contols the minimum number of 16-byte words of a tile to accumulate in a relay stream before sending it off to the destination.\n// If the size of the tile is less than or equal to PARTIAL_SEND_WORDS_THR, then this feild is ignored.\n// Default is 16 words\n"
    }
    ,{
        "MSG_LOCAL_STREAM_CLEAR_NUM",
        0,
        16,
        ""
    }
    ,{
        "MSG_GROUP_STREAM_CLEAR_TYPE",
        (MSG_LOCAL_STREAM_CLEAR_NUM+MSG_LOCAL_STREAM_CLEAR_NUM_WIDTH),
        1,
        ""
    }
    ,{
        "DEBUG_STATUS_STREAM_ID_SEL",
        0,
        STREAM_ID_WIDTH,
        ""
    }
    ,{
        "DISABLE_DEST_READY_TABLE",
        (DEBUG_STATUS_STREAM_ID_SEL+DEBUG_STATUS_STREAM_ID_SEL_WIDTH),
        1,
        ""
    }
    ,{
        "REMOTE_DEST_WORDS_FREE",
        0,
        MEM_WORD_ADDR_WIDTH,
        ""
    }
    ,{
        "BLOB_NEXT_AUTO_CFG_DONE_STREAM_ID",
        0,
        STREAM_ID_WIDTH,
        ""
    }
    ,{
        "BLOB_NEXT_AUTO_CFG_DONE_VALID",
        16,
        1,
        ""
    }
    ,{
        "NCRISC_TRANS_EN",
        0,
        1,
        ""
    }
    ,{
        "NCRISC_TRANS_EN_IRQ_ON_BLOB_END",
        (NCRISC_TRANS_EN + NCRISC_TRANS_EN_WIDTH),
        1,
        ""
    }
    ,{
        "NCRISC_CMD_ID",
        (NCRISC_TRANS_EN_IRQ_ON_BLOB_END + NCRISC_TRANS_EN_IRQ_ON_BLOB_END_WIDTH),
        3,
        ""
    }
    ,{
        "NEXT_NRISC_PIC_INT_ON_PHASE",
        (NCRISC_CMD_ID + NCRISC_CMD_ID_WIDTH),
        19,
        "// Kept for compatibility with grayskull, but doesnt not exist anymore in wormhole\n"
    }
    ,{
        "DRAM_FIFO_RD_PTR_WORDS_LO",
        0,
        24,
        ""
    }
    ,{
        "NCRISC_LOOP_COUNT",
        0,
        24,
        ""
    }
    ,{
        "NCRISC_INIT_ENABLE_BLOB_DONE_IRQ",
        0,
        1,
        ""
    }
    ,{
        "NCRISC_INIT_DISABLE_BLOB_DONE_IRQ",
        (NCRISC_INIT_ENABLE_BLOB_DONE_IRQ + NCRISC_INIT_ENABLE_BLOB_DONE_IRQ_WIDTH),
        1,
        ""
    }
    ,{
        "DRAM_FIFO_RD_PTR_WORDS_HI",
        0,
        4,
        ""
    }
    ,{
        "DRAM_FIFO_WR_PTR_WORDS_LO",
        (DRAM_FIFO_RD_PTR_WORDS_HI + DRAM_FIFO_RD_PTR_WORDS_HI_WIDTH),
        20,
        ""
    }
    ,{
        "NCRISC_TOTAL_LOOP_ITER",
        0,
        24,
        ""
    }
    ,{
        "DRAM_FIFO_WR_PTR_WORDS_HI",
        0,
        8,
        ""
    }
    ,{
        "DRAM_FIFO_CAPACITY_PTR_WORDS_LO",
        (DRAM_FIFO_WR_PTR_WORDS_HI + DRAM_FIFO_WR_PTR_WORDS_HI_WIDTH),
        16,
        ""
    }
    ,{
        "NCRISC_LOOP_INCR",
        0,
        16,
        ""
    }
    ,{
        "NCRISC_LOOP_BACK_NUM_CFG_REG_WRITES",
        (NCRISC_LOOP_INCR+NCRISC_LOOP_INCR_WIDTH),
        8,
        ""
    }
    ,{
        "DRAM_FIFO_CAPACITY_PTR_WORDS_HI",
        0,
        12,
        ""
    }
    ,{
        "DRAM_FIFO_BASE_ADDR_WORDS_LO",
        (DRAM_FIFO_CAPACITY_PTR_WORDS_HI + DRAM_FIFO_CAPACITY_PTR_WORDS_HI_WIDTH),
        12,
        ""
    }
    ,{
        "NCRISC_LOOP_BACK_AUTO_CFG_PTR",
        0,
        24,
        ""
    }
    ,{
        "DRAM_FIFO_BASE_ADDR_WORDS_HI",
        0,
        16,
        ""
    }
    ,{
        "DRAM_EN_BLOCKING",
        (DRAM_FIFO_BASE_ADDR_WORDS_HI + DRAM_FIFO_BASE_ADDR_WORDS_HI_WIDTH),
        1,
        "// Processes the read or write operation to completeion without processing other dram streams in the meantime\n"
    }
    ,{
        "DRAM_DATA_STRUCTURE_IS_LUT",
        (DRAM_EN_BLOCKING + DRAM_EN_BLOCKING_WIDTH),
        1,
        "// Fifo structure in dram holds a dram pointer and size that is used as indirection to a tile in dram\n"
    }
    ,{
        "DRAM_RESET_RD_PTR_TO_BASE_ON_EMPTY",
        (DRAM_DATA_STRUCTURE_IS_LUT + DRAM_DATA_STRUCTURE_IS_LUT_WIDTH),
        1,
        "// During a dram read, if its detected that the fifo is empty the ncrisc will reset the read pointer back to base\n// Its expected that there is no host interaction\n"
    }
    ,{
        "DRAM_RESET_WR_PTR_TO_BASE_ON_FULL",
        (DRAM_RESET_RD_PTR_TO_BASE_ON_EMPTY + DRAM_RESET_RD_PTR_TO_BASE_ON_EMPTY_WIDTH),
        1,
        "// During a dram write, if its detected that the fifo is full the ncrisc will reset the write pointer back to base. Old data will be overwritten.\n// Its expected that there is no host interaction\n"
    }
    ,{
        "DRAM_NO_PTR_UPDATE_ON_PHASE_END",
        (DRAM_RESET_WR_PTR_TO_BASE_ON_FULL + DRAM_RESET_WR_PTR_TO_BASE_ON_FULL_WIDTH),
        1,
        "// The internal ncrisc rd/wr pointers will not be updated at phase end\n// Its expected that there is no host interaction\n"
    }
    ,{
        "DRAM_WR_BUFFER_FLUSH_AND_RST_PTRS",
        (DRAM_NO_PTR_UPDATE_ON_PHASE_END + DRAM_NO_PTR_UPDATE_ON_PHASE_END_WIDTH),
        1,
        "// Before ending the phase the ncrisc will wait until the host has emptied the write buffer and then reset the read and write pointers to base\n// This can be used for hosts that do not want to track wrapping\n// The host must be aware of this behaviour for this functionality to work\n"
    }
    ,{
        "NCRISC_LOOP_NEXT_PIC_INT_ON_PHASE",
        0,
        20,
        ""
    }
};

const std::unordered_map<std::string, std::uint32_t> OLP::fields_by_name = {
    {"STREAM_REMOTE_SRC_X", 0}
    ,{"STREAM_REMOTE_SRC_Y", 1}
    ,{"REMOTE_SRC_STREAM_ID", 2}
    ,{"STREAM_REMOTE_SRC_DEST_INDEX", 3}
    ,{"DRAM_READS__TRANS_SIZE_WORDS_LO", 4}
    ,{"DRAM_READS__SCRATCH_1_PTR", 5}
    ,{"DRAM_READS__TRANS_SIZE_WORDS_HI", 6}
    ,{"STREAM_REMOTE_DEST_X", 7}
    ,{"STREAM_REMOTE_DEST_Y", 8}
    ,{"STREAM_REMOTE_DEST_STREAM_ID", 9}
    ,{"STREAM_LOCAL_DEST_MSG_CLEAR_NUM", 10}
    ,{"STREAM_LOCAL_DEST_STREAM_ID", 11}
    ,{"DRAM_WRITES__SCRATCH_1_PTR_LO", 12}
    ,{"REMOTE_DEST_BUF_SIZE_WORDS", 13}
    ,{"DRAM_WRITES__SCRATCH_1_PTR_HI", 14}
    ,{"INCOMING_DATA_NOC", 15}
    ,{"OUTGOING_DATA_NOC", 16}
    ,{"REMOTE_SRC_UPDATE_NOC", 17}
    ,{"LOCAL_SOURCES_CONNECTED", 18}
    ,{"SOURCE_ENDPOINT", 19}
    ,{"REMOTE_SOURCE", 20}
    ,{"RECEIVER_ENDPOINT", 21}
    ,{"LOCAL_RECEIVER", 22}
    ,{"REMOTE_RECEIVER", 23}
    ,{"PHASE_AUTO_CONFIG", 24}
    ,{"PHASE_AUTO_ADVANCE", 25}
    ,{"DATA_AUTO_SEND", 26}
    ,{"NEXT_PHASE_SRC_CHANGE", 27}
    ,{"NEXT_PHASE_DEST_CHANGE", 28}
    ,{"DATA_BUF_NO_FLOW_CTRL", 29}
    ,{"DEST_DATA_BUF_NO_FLOW_CTRL", 30}
    ,{"REMOTE_SRC_IS_MCAST", 31}
    ,{"NO_PREV_PHASE_OUTGOING_DATA_FLUSH", 32}
    ,{"UNICAST_VC_REG", 33}
    ,{"REG_UPDATE_VC_REG", 34}
    ,{"STREAM_MCAST_END_X", 35}
    ,{"STREAM_MCAST_END_Y", 36}
    ,{"STREAM_MCAST_EN", 37}
    ,{"STREAM_MCAST_LINKED", 38}
    ,{"STREAM_MCAST_VC", 39}
    ,{"STREAM_MCAST_NO_PATH_RES", 40}
    ,{"STREAM_MCAST_XY", 41}
    ,{"MSG_ARB_GROUP_SIZE", 42}
    ,{"MSG_SRC_IN_ORDER_FWD", 43}
    ,{"MSG_HEADER_WORD_CNT_OFFSET", 44}
    ,{"MSG_HEADER_WORD_CNT_BITS", 45}
    ,{"SOURCE_ENDPOINT_NEW_MSG_ADDR", 46}
    ,{"SOURCE_ENDPOINT_NEW_MSG_SIZE", 47}
    ,{"SOURCE_ENDPOINT_NEW_MSGS_NUM", 48}
    ,{"SOURCE_ENDPOINT_NEW_MSGS_TOTAL_SIZE", 49}
    ,{"PHASE_READY_DEST_NUM", 50}
    ,{"PHASE_READY_NUM", 51}
    ,{"PHASE_READY_MCAST", 52}
    ,{"PHASE_READY_TWO_WAY_RESP", 53}
    ,{"STREAM_REMOTE_RDY_SRC_X", 54}
    ,{"STREAM_REMOTE_RDY_SRC_Y", 55}
    ,{"REMOTE_RDY_SRC_STREAM_ID", 56}
    ,{"REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM", 57}
    ,{"REMOTE_DEST_BUF_WORDS_FREE_INC", 58}
    ,{"WAIT_SW_PHASE_ADVANCE_SIGNAL", 59}
    ,{"WAIT_PREV_PHASE_DATA_FLUSH", 60}
    ,{"MSG_FWD_ONGOING", 61}
    ,{"STREAM_CURR_STATE", 62}
    ,{"PHASE_NUM_INCR", 63}
    ,{"CURR_PHASE_NUM_MSGS", 64}
    ,{"NEXT_PHASE_NUM_CFG_REG_WRITES", 65}
    ,{"CLOCK_GATING_EN", 66}
    ,{"CLOCK_GATING_HYST", 67}
    ,{"PARTIAL_SEND_WORDS_THR", 68}
    ,{"MSG_LOCAL_STREAM_CLEAR_NUM", 69}
    ,{"MSG_GROUP_STREAM_CLEAR_TYPE", 70}
    ,{"DEBUG_STATUS_STREAM_ID_SEL", 71}
    ,{"DISABLE_DEST_READY_TABLE", 72}
    ,{"REMOTE_DEST_WORDS_FREE", 73}
    ,{"BLOB_NEXT_AUTO_CFG_DONE_STREAM_ID", 74}
    ,{"BLOB_NEXT_AUTO_CFG_DONE_VALID", 75}
    ,{"NCRISC_TRANS_EN", 76}
    ,{"NCRISC_TRANS_EN_IRQ_ON_BLOB_END", 77}
    ,{"NCRISC_CMD_ID", 78}
    ,{"NEXT_NRISC_PIC_INT_ON_PHASE", 79}
    ,{"DRAM_FIFO_RD_PTR_WORDS_LO", 80}
    ,{"NCRISC_LOOP_COUNT", 81}
    ,{"NCRISC_INIT_ENABLE_BLOB_DONE_IRQ", 82}
    ,{"NCRISC_INIT_DISABLE_BLOB_DONE_IRQ", 83}
    ,{"DRAM_FIFO_RD_PTR_WORDS_HI", 84}
    ,{"DRAM_FIFO_WR_PTR_WORDS_LO", 85}
    ,{"NCRISC_TOTAL_LOOP_ITER", 86}
    ,{"DRAM_FIFO_WR_PTR_WORDS_HI", 87}
    ,{"DRAM_FIFO_CAPACITY_PTR_WORDS_LO", 88}
    ,{"NCRISC_LOOP_INCR", 89}
    ,{"NCRISC_LOOP_BACK_NUM_CFG_REG_WRITES", 90}
    ,{"DRAM_FIFO_CAPACITY_PTR_WORDS_HI", 91}
    ,{"DRAM_FIFO_BASE_ADDR_WORDS_LO", 92}
    ,{"NCRISC_LOOP_BACK_AUTO_CFG_PTR", 93}
    ,{"DRAM_FIFO_BASE_ADDR_WORDS_HI", 94}
    ,{"DRAM_EN_BLOCKING", 95}
    ,{"DRAM_DATA_STRUCTURE_IS_LUT", 96}
    ,{"DRAM_RESET_RD_PTR_TO_BASE_ON_EMPTY", 97}
    ,{"DRAM_RESET_WR_PTR_TO_BASE_ON_FULL", 98}
    ,{"DRAM_NO_PTR_UPDATE_ON_PHASE_END", 99}
    ,{"DRAM_WR_BUFFER_FLUSH_AND_RST_PTRS", 100}
    ,{"NCRISC_LOOP_NEXT_PIC_INT_ON_PHASE", 101}
};
