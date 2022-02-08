function make_dm_management_html(do_info) {
    let div_blocks = $('<div>');
    div_blocks.append($('<label>', {'id': 'manage-dm-name','class': 'dm-manage-title', 'text': do_info.dm_name, 'dm_id': do_info.dm_id }));
    
    let df_content = $('<table>');
    let idf_title = $('<div>', {'class': 'dm-manage-row', 'text': 'Input Device Features'});

    let odf_title = $('<div>', {'class': 'dm-manage-row', 'text': 'Output Device Features'});

    let idf_content = $('<div>', {'class': 'dm-manage-row', });
    let odf_content = $('<div>', {'class': 'dm-manage-row', });

    let sim_flag = false;

    if (!do_info.plural) { // no multi df
        $.each(do_info.df_list, function (index, df){
            let is_checked = ('do' in do_info && do_info.do.dfo.indexOf(df.df_name) != -1);
            let container;
            if (df.df_type == 'input') {
                container = idf_content;
                sim_flag = true;
            } else {
                container = odf_content;
            }

            container.append($('<input>', {'type': 'checkbox', 'class': 'df-select', 'id': df.df_name, 'df_name': df.df_name, 'df_type': df.df_type, 'checked': is_checked}));
            container.append($('<label>', {'text': df.df_name, 'class': 'df-name', 'for': df.df_name }));
            container.append($('<br>'));
        });
    } else { // multi df
        let idf_list = [];
        let odf_list = [];
        let idf_nums = {};
        let odf_nums = {};
        let df_select = {};

        //parse df_name
        $.each(do_info.df_list, function(index, df) {
            let df_list;
            let df_nums;
            if (df.df_type == 'input') {
                df_list = idf_list;
                df_nums = idf_nums;
            } else {
                df_list = odf_list;
                df_nums = odf_nums;
            }

            if (isNaN(df.df_name.substr(-1,1))) {
                df_list.push(df.df_name);
                df_nums[df.df_name] = 1;
                df_select[df.df_name] = 0;
            }
            else {
                df_name = df.df_name.replace(/\d*$/, '');
                df_num = df.df_name.replace(/^\D*/,'')
                if (df_list.indexOf(df_name) == -1) {
                    df_list.push(df_name);
                    df_nums[df_name] = df_num;
                    df_select[df_name] = 0;
                }
                else if (df_nums[df_name] < df_num) {
                    df_nums[df_name] = df_num;
                }
            }
        });

        if ('do' in do_info) {
            $.each(do_info.do.dfo, function(index, df_name) {
                if (isNaN(df_name.substr(-1,1))) {
                    df_select[df_name] = 1
                } else {
                    _df_name = df_name.replace(/\d*$/, '');
                    df_num = df_name.replace(/^\D*/,'')
                    if (df_select[_df_name] < df_num) {
                        df_select[_df_name] = df_num;
                    }
                }
            });
        }

        //generate html codes for multiple dfs
        idf_list.sort().forEach(function multi(df_name, index) {
            idf_content.append($('<label>', {'text': df_name, 'class': 'df-name',}));

            let select = $('<select>', {'class': 'df-select', 'df_name': df_name, 'df_type': 'input', 'multi': idf_nums[df_name]});
            for (i = 0; i <= idf_nums[df_name]; ++i) {
                if (df_select[df_name] == i){
                    select.append($('<option>', {'text': i, 'selected': ''}));
                } else {
                    select.append($('<option>', {'text': i}));
                }
            }
            idf_content.append(select);
            idf_content.append($('<br>'));
            sim_flag = true;
        });

        odf_list.sort().forEach(function multi(df_name, index) {
            odf_content.append($('<label>', {'text': df_name, 'class': 'df-name',}));

            let select = $('<select>', {'class': 'df-select', 'df_name': df_name, 'df_type': 'onput', 'multi':idf_nums[df_name]});
            for (i = 0; i <= odf_nums[df_name]; ++i) {
                if (df_select[df_name] == i) {
                    select.append($('<option>', {'text': i, 'selected': ''}));
                } else {
                    select.append($('<option>', {'text': i}));
                }
            }
            odf_content.append(select);
            odf_content.append($('<br>'));
        });
    }
    df_content.append(idf_title);
    df_content.append(idf_content);
    df_content.append(odf_title);
    df_content.append(odf_content);

    div_blocks.append(df_content);
    if ('do' in do_info) {
        div_blocks.append($('<button>', {'id':'save-dm','text':'Save', 'do_id': do_info.do.do_id}));
        div_blocks.append($('<button>', {'id':'delete-dm','text':'Delete','do_id': do_info.do.do_id}));

        if(simtalk && sim_flag){
            div_blocks.append($('<button>', {'id':'sim-setup','text':'Extra Setup', 'do_id': do_info.do.do_id}));
        }
        if (do_info.do.extra_setup_webpage) {
            div_blocks.append($('<button>', {'id':'extra-setup',
                                             'text':'Extra Setup',
                                             'url': do_info.do.extra_setup_webpage,
                                             'do_id': do_info.do.do_id}));
        }
        div_blocks.append($('<button>', {'id':'sa-gen-setup','text':'Save and Create SA Code','do_id': do_info.do.do_id}));
    } else {
        div_blocks.append($('<button>', {'id':'save-dm','text':'Save',}));
    }

    return div_blocks;
}

function make_model_block_html(do_info, type){
    let do_blocks = $('<div>', {'class': 'do-container'});

    let do_header = $('<div>', {'class': 'do-header'});

    let do_setting = $('<div>', {'class': 'do-setting'});
    do_setting.append($('<img>', {
            'src':'/static/images/setting.png',
            'do_id': do_info.do_id,
            'class':'do-setting-img',
    }));

    let do_device = $('<div>', {'class': 'do-device'});
    let do_device_name;
    if (do_info.d_name) {
        do_device_name = $('<span>', {'class': 'device-name mounted', 'd_id': do_info.d_id, 'text': do_info.d_name});
        if (do_info.status == 'offline') {
            do_device_name.addClass('offline');
        }
    }
    else {
        do_device_name = $('<span>', {'class': 'device-name', 'text': do_info.dm_name});
    }
    do_device.append(do_device_name);

    do_header.append(do_setting);
    do_header.append(do_device);



    let do_content = $('<div>', {'class': 'do-content'});
    $.each(do_info.dfo, function (index, dfo) {
        let dfo_container = $('<div>', {'class': 'dfo-container', 'dfo_id': dfo.dfo_id});
        dfo_container.append($('<label>', { 'class': 'dfo-name', 'text': dfo.alias_name}));
        if (dfo.src) {
            dfo_container.append($('<img>', {
                'class': 'dfo-image',
                'df_type': dfo.df_type,
                'src': dfo.src,
            }));
        }
        do_content.append(dfo_container);
    });

    do_blocks.append(do_header);
    do_blocks.append(do_content);

    return do_blocks;
}

function make_df_management_html(df_info) {
    let div_blocks = $('<div>', {'class': 'df-manage-container', 'dfo_id': df_info.dfo_id});

    let table = $('<table>');

    // header
    let thead = $('<thead>');
    if (df_info.df_type == 'input') {
        thead.append($('<td>', {'colspan': 5, 'text': df_info['dm_name']}));
    } else {
        thead.append($('<td>', {'colspan': 5, 'text': df_info['dm_name']}));
    }
    table.append(thead);

    let tbody = $('<tbody>');

    // Title row
    let tr_title = $('<tr>');

    if (df_info.df_type == 'input') {
        let td_name = $('<td>', {'id':'alias-name-header', 'rowspan': 2});
        td_name.append($('<span>', {'id': 'alias-name', 'text': df_info['alias_name']}));
        td_name.append($('<span>', {'class':'glyphicon glyphicon-pencil edit-alias-name'}));
        tr_title.append(td_name);
        tr_title.append($('<td>', {'text': 'Type', 'rowspan': 2}));
        tr_title.append($('<td>', {'text': 'Function', 'rowspan': 2}));
        tr_title.append($('<td>', {'text': 'Simulation Range', 'colspan': 2}))

        tr_title2 = $('<tr>');
        tr_title2.append($('<td>', {'text': 'Min', 'style': 'border-left: 2px solid'}));
        tr_title2.append($('<td>', {'text': 'Max'}));
        tbody.append(tr_title);
        tbody.append(tr_title2);
    } else {
        let td_name = $('<td>', {'id':'alias-name-header'});
        td_name.append($('<span>', {'id': 'alias-name', 'text': df_info['alias_name']}));
        td_name.append($('<span>', {'class':'glyphicon glyphicon-pencil edit-alias-name'}));
        tr_title.append(td_name);
        tr_title.append($('<td>', {'text': 'Function'}));
        tr_title.append($('<td>', {'text': 'Min'}));
        tr_title.append($('<td>', {'text': 'Max'}));
        tr_title.append($('<td>', {'text': 'Normalization'}))
        tbody.append(tr_title);
    }

    // dfp row
    if (df_info.df_type == 'input') {
        // idf
        for (let idx = 0; idx < df_info['df_parameter'].length; ++idx) {
            let dfp = df_info['df_parameter'][idx];
            let tr = $('<tr>');

            // name
            tr.append($('<td>', {'text': (df_info['df_type']=='input'?'x':'y') + (idx + 1)}));

            // type
            let td_type = $('<td>');
            let type_select = $('<select>', {'class': 'df-type-select', 'val': dfp['idf_type']});
            type_select.append($('<option>', { 'text': 'variant', }));
            type_select.append($('<option>', { 'text': 'sample', }));
            type_select.val(dfp['idf_type']);
            td_type.append(type_select);
            tr.append(td_type);

            // func
            if (idx == 0) {
                let td_func = $('<td>', {'rowspan': df_info['df_parameter'].length});
                let func_select = $('<select>', {'class': 'df-func-select', 'val': dfp['fn_id'], 'height': ((37 * df_info['df_parameter'].length) - 3) + 'px'});

                func_select.append($('<option>', {'fn_id': 0, 'text': 'add new function'}));
                func_select.append($('<option>', {'fn_id': null, 'text': 'disable'}));
                let fn_selected = false;
                for (let idx2 = 0; idx2 < df_info['df_mapping_func'].length; ++idx2) {
                    let func_info = df_info['df_mapping_func'][idx2];
                    if (dfp['fn_id'] == func_info['fn_id']) {
                        fn_selected = true;
                        func_select.append($('<option>', {'fn_id': func_info['fn_id'], 'text': func_info['fn_name'], 'selected': true}));
                    } else {
                        func_select.append($('<option>', {'fn_id': func_info['fn_id'], 'text': func_info['fn_name']}));
                    }
                }
                if (!fn_selected){
                    func_select.children()[1].setAttribute('selected', true);
                }

                td_func.append(func_select);
                tr.append(td_func);
            }

            // min
            let td_min = $('<td>');
            td_min.append($('<input>', {'type': 'text', 'class': 'df-min-input', 'value': dfp['min']}));
            tr.append(td_min);

            // max
            let td_max = $('<td>');
            td_max.append($('<input>', {'type': 'text', 'class': 'df-max-input', 'value': dfp['max']}));
            tr.append(td_max);

            tbody.append(tr);
        }
    } else {
        // odf
        for (let idx = 0; idx < df_info['df_parameter'].length; ++idx) {
            let dfp = df_info['df_parameter'][idx];
            let tr = $('<tr>');

            // name
            tr.append($('<td>', {'text': (df_info['df_type']=='input'?'x':'y') + (idx + 1)}));

            // func
            let td_func = $('<td>');
            let func_select = $('<select>', {'class': 'df-func-select'});

            func_select.append($('<option>', {'fn_id': 0, 'text': 'add new function'}));
            func_select.append($('<option>', {'fn_id': null, 'text': 'disable'}));
            let fn_selected = false;
            for (let idx2 = 0; idx2 < df_info.df_mapping_func.length; ++idx2) {
                let func_info = df_info.df_mapping_func[idx2];
                if (dfp.fn_id == func_info.fn_id) {
                        fn_selected = true;
                    func_select.append($('<option>', {'fn_id': func_info.fn_id, 'text': func_info.fn_name, 'selected': true}));
                } else {
                    func_select.append($('<option>', {'fn_id': func_info.fn_id, 'text': func_info.fn_name}));
                }
            }
            if (!fn_selected){
                func_select.children()[1].setAttribute('selected', true);
            }

            td_func.append(func_select);
            tr.append(td_func);

            // min
            let td_min = $('<td>');
            td_min.append($('<input>', {'type': 'text', 'class': 'df-min-input', 'value': dfp['min']}));
            tr.append(td_min);

            // max
            let td_max = $('<td>');
            td_max.append($('<input>', {'type': 'text', 'class': 'df-max-input', 'value': dfp['max']}));
            tr.append(td_max);

            // normalization
            let td_norm = $('<td>');
            let norm_select = $('<select>', {'class': 'df-norm-select', 'name': 'normalization'});
            norm_select.append($('<option>', {'text': 'disabled', 'value': 0, 'selected': !dfp.normalization}));
            norm_select.append($('<option>', {'text': 'enabled', 'value': 1, 'selected': dfp.normalization}));
            td_norm.append(norm_select);
            tr.append(td_norm);


            tbody.append(tr);
        }
    }

    table.append(tbody);

    div_blocks.append(table);
    div_blocks.append($('<button>', {'id':'save-df','text':'Save', 'dfo_id': df_info.dfo_id}));

    return div_blocks;
}

function make_join_management_html(na_info) {
    let div_blocks = $('<div>', {'class': 'join-manage-container'});

    div_blocks.append($('<div>', {'class': 'join-manage-title', 'text': 'Connection Name: '}));
    div_blocks.append($('<input>', {'type': 'text', 'class': 'join-name', 'value': na_info.na_name}));
    div_blocks.append($('<button>', {'class': 'join-save', 'text': 'Save', 'na_id': na_info.na_id}));
    div_blocks.append($('<button>', {'class': 'join-delete', 'text': 'Delete', 'na_id': na_info.na_id}));
    div_blocks.append($('<hr>'));

    //idf
    $.each(na_info.input, function (index, dfm) {
        let dfm_container = $('<table>', {'class': 'dfm-container', 'dfo_id': dfm.dfo_id});
        let dfm_header = $('<thead>');
        let dfm_header_cell = $('<td>', {'colspan': 3, 'text': dfm.dm_name + ' (IDF)'});
        dfm_header_cell.append($('<button>', {'class': 'dfm-delete', 'text': 'Delete', 'dfo_id': dfm.dfo_id}));
        dfm_header.append(dfm_header_cell);
        dfm_container.append(dfm_header);

        let dfmp_header = $('<tr>');
        dfmp_header.append($('<td>', {'text': dfm.alias_name}));
        dfmp_header.append($('<td>', {'text': 'Type'}));
        dfmp_header.append($('<td>', {'text': 'Function'}));
        dfm_container.append(dfmp_header);

        $.each(dfm.dfmp, function (index, dfmp) {
            let dfmp_row = $('<tr>', {'class': 'dfm-data-row'});
            let dfmp_name_cell = $('<td>', {'text': 'x' + (index + 1)});
            dfmp_row.append(dfmp_name_cell);

            let dfmp_type_cell = $('<td>');
            let dfmp_type_select = $('<select>', {'class': 'dfm-select', 'name': 'input_type'});
            dfmp_type_select.append($('<option>', {'text': 'sample', 'selected': ('sample' == dfmp.idf_type)}));
            dfmp_type_select.append($('<option>', {'text': 'variant', 'selected': ('variant' == dfmp.idf_type)}));
            dfmp_type_cell.append(dfmp_type_select);
            dfmp_row.append(dfmp_type_cell);

            if (0 == index) {
                let dfmp_func_cell = $('<td>', {'rowspan': dfm.dfmp.length});
                let dfmp_func_select = $('<select>', {'class': 'dfm-select', 'name': 'function', 'height': ((35 * dfm.dfmp.length) - 2) + 'px'});
                dfmp_func_select.append($('<option>', {'fn_id': 0, 'text': 'add new function'}));
                dfmp_func_select.append($('<option>', {'fn_id': null, 'text': 'disable', 'selected': true}));
                $.each(dfm.fn_list, function(index2, fn_info) {
                    dfmp_func_select.append($('<option>', {'fn_id': fn_info.fn_id, 'text': fn_info.fn_name, 'selected': (fn_info.fn_id == dfmp.fn_id)}));
                });
                dfmp_func_cell.append(dfmp_func_select);
                dfmp_row.append(dfmp_func_cell);
            }

            dfm_container.append(dfmp_row);
        });

        div_blocks.append(dfm_container);
        div_blocks.append($('<hr>'))
    });

    // multiple input
    if (na_info.multiple.length > 1) {
        let multiple_container = $('<table>', {'class': 'dfm-container'});
        let multiple_header = $('<thead>');
        multiple_header.append($('<td>', {'text': 'Input'}));
        multiple_header.append($('<td>', {'text': 'IDF (Line)'}));
        multiple_header.append($('<td>', {'text': 'Join Function'}));
        multiple_container.append(multiple_header);

        $.each(na_info.multiple, function (index, multiple) {
            let multiple_row = $('<tr>');
            let multiple_name_cell = $('<td>', {'text': 'z' + (index + 1)});
            multiple_row.append(multiple_name_cell);

            let multiple_idx_cell = $('<td>');
            let multiple_idx_select = $('<select>', {'class': 'dfm-select', 'disabled': 'disabled'});
            $.each(na_info.input, function(index2, dfm) {
                multiple_idx_select.append($('<option>', {'dfo_id': dfm.dfo_id, 'text': dfm.alias_name, 'selected': (dfm.dfo_id == multiple.dfo_id)}));
            });
            multiple_idx_cell.append(multiple_idx_select);
            multiple_row.append(multiple_idx_cell);

            if (0 == index) {
                let multiple_func_cell = $('<td>', {'rowspan': na_info.multiple.length});
                let multiple_func_select = $('<select>', {'class': 'dfm-select', 'id': 'join-function-select', 'height': ((35 * na_info.multiple.length) - 2) + 'px'});
                multiple_func_select.append($('<option>', {'fn_id': 0, 'text': 'add new function'}));
                multiple_func_select.append($('<option>', {'fn_id': null, 'text': 'disable', 'selected': true}));
                $.each(na_info.fn_list, function(index2, fn_info) {
                    multiple_func_select.append($('<option>', {'fn_id': fn_info.fn_id, 'text': fn_info.fn_name, 'selected': (fn_info.fn_id == multiple.fn_id)}));
                });
                multiple_func_cell.append(multiple_func_select);
                multiple_row.append(multiple_func_cell);
            }

            multiple_container.append(multiple_row);
        });

        div_blocks.append(multiple_container);
        div_blocks.append($('<hr>'));
    }

    //odf
    $.each(na_info.output, function (index, dfm) {
        let dfm_container = $('<table>', {'class': 'dfm-container', 'dfo_id': dfm.dfo_id});
        let dfm_header = $('<thead>');
        let dfm_header_cell = $('<td>', {'colspan': 3, 'text': dfm.dm_name + ' (ODF)'});
        dfm_header_cell.append($('<button>', {'class': 'dfm-delete', 'text': 'Delete', 'dfo_id': dfm.dfo_id}));
        dfm_header.append(dfm_header_cell);
        dfm_container.append(dfm_header);

        let dfmp_header = $('<tr>');
        dfmp_header.append($('<td>', {'text': dfm.alias_name}));
        dfmp_header.append($('<td>', {'text': 'Function'}));
        dfmp_header.append($('<td>', {'text': 'Normalization'}));
        dfm_container.append(dfmp_header);

        $.each(dfm.dfmp, function (index, dfmp) {
            let dfmp_row = $('<tr>', {'class': 'dfm-data-row'});
            let dfmp_name_cell = $('<td>', {'text': 'y' + (index + 1)});
            dfmp_row.append(dfmp_name_cell);

            let dfmp_func_cell = $('<td>');
            let dfmp_func_select = $('<select>', {'class': 'dfm-select', 'name': 'function'});
            dfmp_func_select.append($('<option>', {'fn_id': 0, 'text': 'add new function'}));
            dfmp_func_select.append($('<option>', {'fn_id': null, 'text': 'disable', 'selected': true}));
            $.each(dfm.fn_list, function(index2, fn_info) {
                dfmp_func_select.append($('<option>', {'fn_id': fn_info.fn_id, 'text': fn_info.fn_name, 'selected': (fn_info.fn_id == dfmp.fn_id)}));
            });
            dfmp_func_cell.append(dfmp_func_select);
            dfmp_row.append(dfmp_func_cell);

            let dfmp_norm_cell = $('<td>');
            let dfmp_norm_select = $('<select>', {'class': 'dfm-select', 'name': 'normalization'});
            dfmp_norm_select.append($('<option>', {'text': 'disabled', 'value': 0, 'selected': !dfmp.normalization}));
            dfmp_norm_select.append($('<option>', {'text': 'enabled', 'value': 1, 'selected': dfmp.normalization}));
            dfmp_norm_cell.append(dfmp_norm_select);
            dfmp_row.append(dfmp_norm_cell);

            dfm_container.append(dfmp_row);
        });

        div_blocks.append(dfm_container);
        div_blocks.append($('<hr>'))
    });

    return div_blocks;
}

function make_join_monitor_html(data) {
    let div_blocks = $('<div>');

    // idf
    let table_idf = $('<table>');

    let thead_idf = $('<thead>');
    thead_idf.append($('<td>', {'text': 'IDF Monitor', 'colspan': data.input.length}));
    table_idf.append(thead_idf);

    let tbody_idf = $('<tbody>');
    let idf_title = $('<tr>');
    for (let idx=0; idx < data.input.length; ++idx) {
        let idf = data.input[idx];
        if (idx == 0)
            idf_title.append($('<td>', {'class': 'monitor-name idf active', 'dfo_id': idf.dfo_id, 'text': idf.alias_name}));
        else
            idf_title.append($('<td>', {'class': 'monitor-name idf', 'dfo_id': idf.dfo_id, 'text': idf.alias_name}));
    }
    tbody_idf.append(idf_title);
    let td_idf = $('<td>', {'colspan': data.input.length});
    for (let idx=0; idx < data.input.length; ++idx) {
        let df = data.input[idx];
        let df_title_container = $('<div>', {'class': 'monitor-container'});
        let df_content_container = $('<div>', {'class': 'monitor-container idf', 'mac_addr': df.mac_addr, 'df_name': df.df_name, 'dfo_id': df.dfo_id});

        if (idx != 0) {
            df_title_container.addClass('disappear-flag');
            df_content_container.addClass('disappear-flag');
        }

        let param_row = $('<div>', {'class': 'monitor-content-row'});
        param_row.append($('<div>', {'class': 'monitor-content', 'text': 'Timestamp'}));
        for (let idx2=0;idx2<df.param_no; ++idx2) {
            param_row.append($('<div>', {'class': 'monitor-content', 'text': 'x' + (idx2 + 1)}));
        }

        df_title_container.append(param_row);

        td_idf.append(df_title_container);
        td_idf.append(df_content_container);
    }
    tbody_idf.append($('<tr>').append(td_idf));
    table_idf.append(tbody_idf);

    div_blocks.append(table_idf);
    div_blocks.append($('<hr>'));

    // multiple join
    if (data.input.length > 1) {
        let table_join = $('<table>');

        let thead_join = $('<thead>');
        thead_join.append($('<td>', {'text': 'MultipleJoin Monitor'}));
        table_join.append(thead_join);

        let tbody_join = $('<tbody>');
        let title_container = $('<div>', {'class': 'monitor-container'});
        let content_container = $('<div>', {'class': 'monitor-container multiple', 'mac_addr': '_join', 'df_name': '_join'});
        let param_row = $('<div>', {'class': 'monitor-content-row'});
        param_row.append($('<div>', {'class': 'monitor-content', 'text': 'Timestamp'}));
        for(let i = 0; i < data.input.length; ++i) {
            param_row.append($('<div>', {'class': 'monitor-content', 'text': 'z' + (i + 1)}));
        }
        title_container.append(param_row);

        let td = $('<td>');
        let tr = $('<tr>');
        td.append(title_container);
        td.append(content_container);
        tr.append(td);
        tbody_join.append(tr);
        table_join.append(tbody_join);

        div_blocks.append(table_join);
        div_blocks.append($('<hr>'));
    }

    // odf
    let table_odf = $('<table>');

    let thead_odf = $('<thead>');
    thead_odf.append($('<td>', {'text': 'ODF Monitor', 'colspan': data.output.length}));
    table_odf.append(thead_odf);

    let tbody_odf = $('<tbody>');
    let odf_title = $('<tr>');
    for (let idx=0; idx < data.output.length; ++idx) {
        let odf = data.output[idx];
        if (idx == 0)
            odf_title.append($('<td>', {'class': 'monitor-name odf active', 'dfo_id': odf.dfo_id, 'text': odf.alias_name}));
        else
            odf_title.append($('<td>', {'class': 'monitor-name odf', 'dfo_id': odf.dfo_id, 'text': odf.alias_name}));
    }
    tbody_odf.append(odf_title);
    let td_odf = $('<td>', {'colspan': data.output.length});
    for (let idx=0; idx < data.output.length; ++idx) {
        let df = data.output[idx];
        let df_title_container = $('<div>', {'class': 'monitor-container'});
        let df_content_container = $('<div>', {'class': 'monitor-container odf', 'mac_addr': df.mac_addr, 'df_name': df.df_name, 'dfo_id': df.dfo_id});

        if (idx != 0) {
            df_title_container.addClass('disappear-flag');
            df_content_container.addClass('disappear-flag');
        }

        let param_row = $('<div>', {'class': 'monitor-content-row'});
        param_row.append($('<div>', {'class': 'monitor-content', 'text': 'Timestamp'}));
        for (let idx2=0;idx2<df.param_no; ++idx2) {
            param_row.append($('<div>', {'class': 'monitor-content', 'text': 'y' + (idx2 + 1)}));
        }

        df_title_container.append(param_row);

        td_odf.append(df_title_container);
        td_odf.append(df_content_container);
    }
    tbody_odf.append($('<tr>').append(td_odf));
    table_odf.append(tbody_odf);

    div_blocks.append(table_odf);
    div_blocks.append($('<hr>'));

    // error msg
    let table_error = $('<table>');

    let thead_error = $('<thead>');
    thead_error.append($('<td>', {'text': 'Error Message'}));
    table_error.append(thead_error);

    let tbody_error = $('<tbody>');
    let container = $('<div>', {'class': 'monitor-container', 'id': 'monitor-error'});
    tbody_error.append($('<tr>').append($('<td>').append(container)));
    table_error.append(tbody_error);

    div_blocks.append(table_error);

    return div_blocks;
}

function make_function_management_html(dfo_fn_info) {
    let div_block = $('<div>', {'class': 'function-manage-container',
                                'df_id': dfo_fn_info.df_id,
                                'dfo_id': dfo_fn_info.dfo_id});
    let div_header = $('<div>', {'class': 'function-manage-header'});
    let div_content = $('<div>', {'class': 'function-manage-content'});

    div_header.append($('<label>', {'class': 'function-manage-title', 'text': 'Function Management'}));
    div_header.append($('<button>', {'id': 'function-manage-close',
                                     'text': 'Close'}));

    // global function list
    let fn_list_column = $('<div>');
    let global_fn_cell = $('<div>', {'class': 'function-cell', 'text': 'Global Function List'});
    let global_fn_select = $('<select>', {'class': 'function-select', 'id': 'global-function-select', 'multiple': true});
    $.each(dfo_fn_info.other_fn_list, function(index, fn_info) {
        global_fn_select.append($('<option>', {'fn_id': fn_info.fn_id, 'text': fn_info.fn_name}));
    });
    global_fn_cell.append($('<br>'));
    global_fn_cell.append(global_fn_select);

    let control_fn_cell = $('<div>', {'class': 'control-function-cell'});
    control_fn_cell.append($('<br>'));
    control_fn_cell.append($('<button>', {'id': 'move-in-function', 'text': '>>>', 'disabled': ''}));
    control_fn_cell.append($('<br>'));
    control_fn_cell.append($('<button>', {'id': 'move-out-function', 'text': '<<<', 'disabled': ''}));

    // df function list
    let df_fn_cell = $('<div>', {'class': 'function-cell', 'text': dfo_fn_info.df_name + ' Function List'});
    let df_fn_select = $('<select>', {'class': 'function-select', 'id': 'df-function-select'});
    df_fn_select.attr('size', 4);
    $.each(dfo_fn_info.fn_list, function(index, fn_info) {
        df_fn_select.append($('<option>', {'fn_id': fn_info.fn_id, 'text': fn_info.fn_name}));
    });
    df_fn_cell.append($('<br>'));
    df_fn_cell.append(df_fn_select);

    fn_list_column.append(global_fn_cell);
    fn_list_column.append(control_fn_cell);
    fn_list_column.append(df_fn_cell);

    div_content.append(fn_list_column);
    div_content.append('Function Name: ');;
    div_content.append($('<input>', {'id': 'function-name', 'class': 'readonly-flag', 'type': 'text'}).attr('readOnly', 'true'));
    div_content.append($('<button>', {'id': 'add-new-function', 'text': 'New'}));
    div_content.append($('<button>', {'id': 'delete-function', 'text': 'delete'}));
    div_content.append($('<button>', {'id': 'save-function', 'text': 'save'}));
    div_content.append($('<textarea>', {'id': 'function-code-area'}));

    div_block.append(div_header);
    div_block.append(div_content);

    return div_block;
}
