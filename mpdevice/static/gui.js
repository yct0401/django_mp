var _is_init = false;
var project_auto_on_timer;
var monitor_topic = null;

jQuery.expr[':'].regex = function(elem, index, match) {
    let matchParams = match[3].split(','),
        validLabels = /^(data|css):/,
        attr = {
            method: matchParams[0].match(validLabels) ?
            matchParams[0].split(':')[0] : 'attr',
            property: matchParams.shift().replace(validLabels,'')
        },
        regexFlags = 'ig',
        regex = new RegExp(matchParams.join('').replace(/^\s+|\s+$/g,''), regexFlags);
    return regex.test(jQuery(elem)[attr.method](attr.property));
}

function anno_callback(msg) {
    console.log('[anno]', msg);
    if (msg.type == 'monitor_anno') {
        monitor_topic = msg['data'];

        if (monitor_topic) { // Subscribe the monitor topic if it is not null
          self.mqtt_client.subscribe(monitor_topic, monitor_callback);
        } else {
          alert(
            'The monitor functionality is not available now, please contact the administrator.'
          );
        }
    } else if (msg.type == 'offline') {
        $('.device-name[d_id=' + msg.data.d_id +']').addClass('offline');
    } else if (msg.type == 'online') {
        $('.device-name[d_id=' + msg.data.d_id +']').removeClass('offline');
    } else if (msg.type == 'deregister') {
        reload_data(null, null, false);
    }
    else {
        //reload_data(null, null, false);
    }
}

function mqtt_connect_callback(status) {
    console.log(status);
    if (status) {
        $('#alert-bar').addClass('hidden');
    } else {
        $('#alert-bar').removeClass('hidden');
    }
}

function gui_init() {
    resize();

    // check pid from url hash
    if (p_id === -1) {
        window.location.hash.substr(1).split(',').forEach((val)=>{
            let l = val.split('=');
            pid = (l[0] == 'pid') ? Number(l[1]) : p_id;
        })
    }

    // load project if p_id given
    if (p_id !== -1) {
        $('h1').remove();
        $('.dm-list-container, #change-project-status, #delete-project, #change-simulator-status, #export-file').removeClass('hidden');
        reload_data(null, null, false);
    }

    if (_is_init) {return;}

    _is_init = true;

    // nav
    $(document).on('click', '#delete-project', delete_project);
    $(document).on('click', '#change-project-status', change_project_status);
    $(document).on('click', '#change-simulator-status', change_simulator_status);

    // nav of subsystems
    $.each($('.dm-list-container'), init_subsystem_dm_nav);

    // nav child
    $(document).on('click', '.project-select', switch_project); // select project
    $(document).on('click', '.dm-select', create_device_object); // select device model

    // device model
    $(document).on('click', '#save-dm', save_device_object); // save device model
    $(document).on('click', '#delete-dm', delete_device_object); //delete device model
    $(document).on('click', '#sim-setup', set_simtalk); // entering SimTalk Setup WebPage
    $(document).on('click', '#extra-setup', extra_setup); // open extra_url for binding device
    $(document).on('click', '#sa-gen-setup', save_and_sa_gen_setup); // entering SimTalk Setup WebPage

    // device object
    $(document).on('click', '.do-setting-img', modify_do); // device object management
    $(document).on('click', '.dfo-container', modify_dfo); // deivce feature object management
    $(document).on('click', '.do-device', bind_device); // bind device
    $(document).on('click', '.choose-device', click_device); // select bind device

    // device feature object
    $(document).on('click', '#save-df', save_df); // save device feature object
    $(document).on('change', '.df-func-select', select_function); //change device feature function
    $(document).on('click', '.edit-alias-name', modify_alias_name); // modify dfo alias name
    $(document).on('focusout', '#tb-alias-name', change_alias_name); // input alias name listener
    $(document).on('keyup', '#tb-alias-name', function (e) { // input alias name listener
        if (e.keyCode == 13) {
            $('#tb-alias-name').blur();
        }
    });

    // join
    $(document).on('click', '.join-image', modify_join); // join management
    $(document).on('contextmenu', '.join-image', monitor_join); // join monitor
    $(document).on('click', '.join-save', save_join); // save join
    $(document).on('click', '.join-delete', delete_join); // delete join
    $(document).on('click', '.dfm-delete', delete_dfm); // delete join link
    $(document).on('change', '.dfm-select[name=function]', select_function); // change link function
    $(document).on('change', '#join-function-select', select_function); // change join function

    // monitor
    $(document).on('click', '.monitor-name', change_monitor_panel);

    // function
    $(document).on('change', '#global-function-select', select_global_function);
    $(document).on('change', '#df-function-select', select_df_function);
    $(document).on('click', '#move-in-function', create_functionSDF);
    $(document).on('click', '#move-out-function', delete_functionSDF);
    $(document).on('click', '#add-new-function', add_new_function);
    $(document).on('click', '#save-function', save_function);
    $(document).on('click', '#delete-function', delete_function);
    $(document).on('click', '#function-manage-close', close_manage_function);

    // resize window
    $(window).resize(function() { redraw_connect_line(); });
}

function create_project() {
    let p_name = "";
    while (true) {
        p_name = prompt("Enter your project name:");
        if (!p_name) {
            return;
        }

        if (!/^[-\d\w]+$/i.test(p_name) || p_name == "add project") {
            alert("Invalid name");
            continue;
        }

        if ($('.project-select a[name="' + p_name + '"]').length) {
            alert('The project name is exist.');
            continue;
        }
        break;
    }

    function callback(data) {
        if ('p_id' in data) {
            // update project list title
            p_id = data['p_id'];
            $('#project-list-header').html(p_name + '<span class="caret"></span>');

            // update project list
            item = $('<li>', {'class':'project-select', 'p_id':p_id});
            item.append($('<a>', {'text':p_name, 'name':p_name}));
            item.appendTo($('#project-list'));

            // reload data
            gui_init();

        } else {
            alert('create new project failed!');
        }
    }
    mqtt_client.request('create_project', {'p_name': p_name}, callback);
}

function delete_project() {
    function callback(data) {
        if ('p_id' in data && data['p_id'] === p_id) {
            window.location = '/connection';
        }
    }

    // If use SimTalk, stop it.
    if ($('#simulator-status-switch').find('input')[0].checked && simtalk) {
        stop_simtalk_simulation();
    }

    mqtt_client.request('delete_project', {'p_id':p_id}, callback);
}

function get_project_list() {
    function callback(data) {
        project_list = $('#project-list');
        project_list.empty();

        item = $('<li>', {'class':'project-select', 'p_id':0});
        item.append($('<a>', {'text':'add project'}));
        item.appendTo(project_list);

        $.each(data, function (index, project) {
            item = $('<li>', {'class':'project-select', 'p_id':project['p_id']});
            item.append($('<a>', {'text':project['p_name'], 'name':project['p_name']}));
            item.appendTo(project_list);
        });
    }
    mqtt_client.request('get_project_list', null, callback);
}

function switch_project() {
    let originalProjectID = p_id;
    p_id = parseInt($(this).attr('p_id'));

    if (p_id !== 0) {
        gui_init();
        $('#project-list-header').html($(this).text() + '<span class="caret"></span>');
    } else {
        // Assign the original project ID back if we are going to create a new project
        p_id = originalProjectID;
        create_project();
    }
}

function change_project_status() {
    clearTimeout(project_auto_on_timer);
    function callback(data) {
        $('#change-project-status').val(data.status == 'on');
        $('#change-project-status').find('.light').css('background', (data.status == 'on') ? 'green' : 'red');
        if (data.status == 'off') {
            project_auto_on_timer = setTimeout(change_project_status, 1000);
        }
    }

    if ($('#change-project-status').val()) {
        mqtt_client.request('update_project', {'p_id':p_id, 'status': 'off'}, callback);
    } else {
        mqtt_client.request('update_project', {'p_id':p_id, 'status': 'on'}, callback);
    }
}

// register child window callback
window.addEventListener('message', (e)=>{
    if (e.data == 'SimTalk start'){
        $('#simulator-status-switch').find('input')[0].checked = true;
    }
});

function stop_simtalk_simulation() {
    let url = `${simtalk}/execution/${u_id}/${p_id}/`;
    fetch(url, {
        method: 'DELETE',
        credentials: 'include', // to include cookies in header, using credentials
    }).then((res)=>{
        $('#simulator-status-switch').find('input')[0].checked = false;
    }).catch(()=>{
        alert('something wrong.');
    });
}

function change_simulator_status() {
    function callback(status) {
        $('#simulator-status-switch').find('input')[0].checked = (status == 'on');
    }

    if ($('#simulator-status-switch').find('input')[0].checked) {
        if (!simtalk) {
            // No SimTalk, use basic simulatioin
            mqtt_client.request('turn_off_simulation', {'p_id':p_id}, callback);
        } else {
            // Use SimTalk simulatioin
            stop_simtalk_simulation();
        }
    } else {
        if (!simtalk) {
            // No SimTalk, use basic simulatioin
            mqtt_client.request('turn_on_simulation', {'p_id':p_id}, callback);
        } else {
            // Use SimTalk simulatioin
            let url = `${simtalk}/execution/${u_id}/${p_id}/`;
            let child = window.open(url, 'SimTalk', "width=955,height=900");
        }
    }
}

function reload_device_model_list() {
    function callback(data) {
        let dm_list = $('#dm-list');
        dm_list.empty();

        $.each(data.dm_list, function (index, dm) {
            let item = $('<li>', {'class':'dm-select', 'dm_id': dm['dm_id']});
            item.append($('<a>', {'text':dm['dm_name']}));
            item.appendTo(dm_list);
        });
    }
    mqtt_client.request('get_device_model_list', null, callback);
}

function show_device_object_management(data) {
    clear_right_window();
    $('#right-window').append(make_dm_management_html(data));

    if (data['df_list'].length == 1 && !('do' in data)) {
        $('.df-select').prop('checked', true);
        $('#save-dm').trigger('click');
    }
}

function create_device_object(){
    let dm_id = $(this).attr('dm_id');
    if (dm_id) {
        mqtt_client.request('get_device_model_info', {'dm_id':dm_id}, show_device_object_management);
    }
}

function handle_save_device_object(do_id) {
    if (!do_id) {
        do_id = 0;
    }
    let data = {
        'p_id': p_id,
        'df': []
    };

    $.each($('input[type="checkbox"].df-select:checked'), function(index, input) {
        data['df'].push($(input).next().text());
    });

    $.each($('select.df-select'), function(index, input) {
        df_name = $(input).prev().text();
        multi = $(input).attr('multi') != 1;
        for (i = 1; i <= $(input).val(); ++i) {
            data['df'].push(df_name +  (multi ? i : ''));
        }
    });

    function callback(data) {
        reload_data(data, do_id=data.do_id);
    }
    if (do_id == 0) {
        data['dm_id'] = $('#manage-dm-name').attr('dm_id');
        mqtt_client.request('create_device_object', data, callback);
    } else {
        data['do_id'] = do_id;
        mqtt_client.request('update_device_object', data, callback);
    }
}

function save_device_object(){
    let do_id = $(this).attr('do_id');
    handle_save_device_object(do_id);
}

function delete_device_object() {
    if (confirm("Are you sure to delete this device object?")) {
        mqtt_client.request('delete_device_object',
                            {'do_id': $(this).attr('do_id')},
                            reload_data);
    }
}

function set_simtalk(){
    let do_id = this.getAttribute('do_id');
    let url = `${simtalk}/setup/${u_id}/${p_id}/${do_id}/`;
    window.open(url, "", "width=955,height=900");
}

function extra_setup() {
    let do_id = this.getAttribute('do_id');
    let d_name = $('.do-setting,img[do_id=' + do_id + ']').parent().siblings().children()[0].innerText;
    window.open(this.getAttribute('url'), d_name, 'height=800,width=700,left=300,top=100');
}

function save_and_sa_gen_setup(){
    let do_id = this.getAttribute('do_id');
    handle_save_device_object(do_id);
    let url = `${sa_gen}/SaGen/${username}/${p_id}/${do_id}/`;
    window.open(url, "", "width=955,height=900");
}

function reload_data(_, do_id=null, clear_right=true) {
    function callback(data) {
        remove_all_data(clear_right);

        $('#change-project-status').val(data.status == 'on');
        $('#change-project-status').find('.light').css('background', (data.status == 'on') ? 'green' : 'red');
        $('#simulator-status-switch').find('input')[0].checked = (data.sim == 'on');
        if (data.status != 'on') {
            project_auto_on_timer = setTimeout(change_project_status, 1000);
        }

        $.each(data['ido'], function(index, ido) {
            $('#in-device-column').append(make_model_block_html(ido));
        });
        $.each(data['odo'], function(index, odo) {
            $('#out-device-column').append(make_model_block_html(odo));
        });
        redraw_connect_line(data);

        //auto bind device
        if (do_id && (typeof(do_id) == 'number')) {
            $('.do-setting-img[do_id=' + do_id + ']').parents('.do-container').find('.do-device').click();
        }
    }
    if (p_id > 0) {
        mqtt_client.request('get_project_info', {'p_id':p_id}, callback);
    }
}

function clear_right_window() {
    $('#right-window').empty();
    if (monitor_topic) {
        self.mqtt_client.unsubscribe(monitor_topic);
        monitor_topic = null;
    }

}

function remove_all_data(clear_right=true) {
    if (clear_right) {
        clear_right_window();
    }

    $('#out-device-column').children().each(function() {$(this).remove();});
    $('#in-device-column').children().each(function() {$(this).remove();});
    $('.join-container').each(function(index ) {
        $(this).removeAttr('na_id');
        $(this).removeClass('');
        $(this).removeClass('join-clicked');
        $(this).addClass('hidden-flag');
    });
    let canvas = $('canvas');
    let ctx = canvas[0].getContext("2d");
    ctx.clearRect(0 , 0 , canvas.width(), canvas.height());
}

function modify_do() {
    let do_id = $(this).attr('do_id');
    mqtt_client.request('get_device_object_info',
                        {'do_id': do_id, 'p_id': p_id},
                        show_device_object_management);
}

function save_df() {
    let alias_name = $('#alias-name').text();
    let dfo_id = $('#save-df').attr('dfo_id');
    let dfp = [];

    for (let i = 0; i < $('.df-min-input').length; ++i) {
        let dfp_tmp = {};
        // df_type
        if ($('.df-type-select').get(i)) {
            dfp_tmp.idf_type = $('.df-type-select option:selected').get(i).value;
        }

        // min
        if ($('.df-min-input').get(i)) {
            dfp_tmp.min = $('.df-min-input').get(i).value;
        }

        // max
        if ($('.df-max-input').get(i)) {
            dfp_tmp.max = $('.df-max-input').get(i).value;
        }

        // fn_id
        if ($('.df-func-select')) {
            let fn_select_length = $('.df-func-select option:selected').length;
            let fn_id = $($('.df-func-select option:selected').get((fn_select_length>i?i:fn_select_length))).attr('fn_id');
            dfp_tmp.fn_id = fn_id ? fn_id : null;
        }

        // normalization
        if ($('.df-norm-select').get(i)) {
            dfp_tmp.normalization = Number($('.df-norm-select').get(i).value);
        }

        dfp.push(dfp_tmp);
    }

    let data = {'alias_name': alias_name, 'dfo_id': dfo_id, 'df_parameter': dfp};
    mqtt_client.request('update_device_feature_object', data, reload_data);
}

function modify_alias_name() {
    $('#alias-name').addClass('disappear-flag');
    $('.edit-alias-name').addClass('disappear-flag');

    $('#alias-name-header').append($('<input>', {'id': 'tb-alias-name', 'value': $('#alias-name').text(), 'type':'text'}));
    $('#tb-alias-name').select();
}

function change_alias_name() {
    let old_name = $('#alias-name').text();
    let new_name = $('#tb-alias-name').val().trim();
    if (new_name && new_name != old_name) {
        $('#alias-name').text(new_name);
    }

    $('#tb-alias-name').remove();
    $('#alias-name, .edit-alias-name').removeClass('disappear-flag');
}

function modify_join() {
    let join_container = $(this).parents('.join-container');
    let na_id = join_container.attr('na_id');
    let dfo_id = $('.clicked').attr('dfo_id');

    if ($(this).hasClass('join-clicked')) { // current join is selected, unselect
        $(this).removeClass('join-clicked');
    } else if ($('.join-clicked').length != 0) { // other join is selected, selected current one
        $('.join-clicked').removeClass('join-clicked');
        $(this).addClass('join-clicked');
        mqtt_client.request('get_na_info', {'na_id': na_id, 'p_id': p_id}, show_join_management);
    } else if (dfo_id) { // dfo is selected, create link
        create_link(dfo_id, na_id);
        $('.clicked').removeClass('clicked');
    } else { // nothing is selected, select self
        $(this).addClass('join-clicked');
        mqtt_client.request('get_na_info', {'na_id': na_id, 'p_id': p_id}, show_join_management);
    }

}

function show_join_management(data) {
    clear_right_window();
    $('#right-window').append(make_join_management_html(data));
    redraw_connect_line();
}

/* monitor block start */
function monitor_join() {
    let na_id = $(this).parents('.join-container').attr('na_id');
    let credential_id = window.sessionStorage.getItem('credential_id')
    $('.clicked').removeClass('clicked');
    $('.join-clicked').removeClass('join-clicked');

    mqtt_client.request(
      'get_na_monitor', {
        'na_id': na_id, 'p_id': p_id, 'aa_credential_id': credential_id,
      },
      show_join_monitor
    );

    return false;
}

function show_join_monitor(data) {
    clear_right_window();
    $('#right-window').append(make_join_monitor_html(data));
    redraw_connect_line();
}

function change_monitor_panel() {
    if ($(this).hasClass('.active')){
        return;
    }

    let dfo_id = $(this).attr('dfo_id');
    let type = ($(this).hasClass('idf') ? 'idf' : 'odf');

    // change active df name
    $('.monitor-name.' + type + '.active').removeClass('active');
    $(this).addClass('active');

    // change active df content
    $('.monitor-container.' + type).addClass('disappear-flag');
    $('.monitor-container.' + type + '[dfo_id=' + dfo_id +']').removeClass('disappear-flag');
}

function monitor_callback(payload) {
    if (payload.op == 'monitor_info') {
        let data = payload.data;
        update_monitor(data.idf.id[0], data.idf.id[1], data.idf.input);
        update_monitor('_join', '_join', data.join.output);
        $.each(data.odf, function (idx, odf_monitor) {
            update_monitor(odf_monitor.id[0], odf_monitor.id[1], odf_monitor.output);
        })
    } else if (payload.op == 'esm_error') {
        let data = payload.data;
        $('#monitor-error').append($('<label>', {'text': 'Error:'}));
        $('#monitor-error').append('<pre>' + data.msg.replace(/\n/g, '<br />') + '</pre>');
        $('#monitor-error').append($('<hr>'));
    }
}

function update_monitor(mac_addr, df_name, data) {
    let data_row = $('<div>', {'class': 'monitor-content-row'});
    data_row.append($('<div>', {'class': 'monitor-content', 'text': window.Date().split(' ')[4]}));
    if( Object.prototype.toString.call(data) === '[object Array]' ) {
        for (let idx=0;idx<data.length;++idx) {
            data_row.append($('<div>', {'class': 'monitor-content', 'text': data[idx]}));
        }
    } else {
        data_row.append($('<div>', {'class': 'monitor-content', 'text': data}));
    }
    if ($('.monitor-container[mac_addr=' + mac_addr + '][df_name="' + df_name + '"]').length) {
        $('.monitor-container[mac_addr=' + mac_addr + '][df_name="' + df_name + '"]').append(data_row);

        // auto scroll to bottom
        $('.monitor-container[mac_addr=' + mac_addr + '][df_name="' + df_name + '"]').scrollTop($('.monitor-container[mac_addr=' + mac_addr + '][df_name="' + df_name + '"]')[0].scrollHeight)
    }
}
/* monitor block end */

function show_df_info_callback(data) {
    clear_right_window();
    $('#right-window').append(make_df_management_html(data));
}

function modify_dfo() {
    let df_name = $(this).children('.dfo-name').text();
    let is_odf = $($(this).parents()[2]).attr('id') == 'out-device-column';
    let dfo_id = $(this).attr('dfo_id');

    if ($(this).hasClass('clicked')) { // current dfo is selected, unselect
        $(this).removeClass('clicked');
    } else if ($('.clicked').length != 0) { // other dfo is selected, check both df_type
        let self_parents_id = $(this).parents(":regex(id, .*-device-column)").attr('id');
        let target_parents_id = $('.clicked').parents(":regex(id, .*-device-column)").attr('id');

        if (self_parents_id == target_parents_id) { // if df_type are the same, selected current dfo
            $('.clicked').removeClass('clicked');
            $(this).addClass('clicked');
            mqtt_client.request('get_device_feature_object_info', {'dfo_id': dfo_id, 'p_id': p_id}, show_df_info_callback);
        } else { // if df_type are different, create new join
            create_join(dfo_id, $('.clicked').attr('dfo_id'));
        }

        $('.clicked').removeClass('clicked');
    } else if ($('.join-clicked').length != 0) { // join is selected, create link
        let na_id = $('.join-clicked').parents('.join-container').attr('na_id');
        create_link(dfo_id, na_id);
        $('.join-clicked').removeClass('join-clicked');
    } else { // nothing is seleted, select self
        $(this).addClass('clicked');
        mqtt_client.request('get_device_feature_object_info', {'dfo_id': dfo_id, 'p_id': p_id}, show_df_info_callback);
    }
}

function create_join(dfo_id1, dfo_id2) {
    function callback(data) {
        if (data){
            mqtt_client.request('get_na_info', {'na_id': data.na_id, 'p_id': p_id}, show_join_management);
        }
    }

    let join_container = get_new_join_container(dfo_id1, dfo_id2);
    let payload = {};
    payload.dfo_ids = [dfo_id1, dfo_id2];
    payload.p_id = p_id;
    payload.na_idx = $('.join-container').index(join_container);
    payload.na_name = 'Join' + (payload.na_idx + 1);

    mqtt_client.request('create_na', payload, callback);
}

function save_join() {
    $('.join-save').text('Saving');
    $('.join-save').attr('disabled', true);
    let na_id = $(this).attr('na_id');
    let na_name = $('.join-name').val();
    let payload = {
        'na_id': na_id,
        'na_name': na_name,
        'dfm_list': [],
        'multiplejoin_fn_id': null,
    };

    //get dfm info
    $('.dfm-container').each(function(index, container) {
        container = $(container);

        let dfm = {
            'dfo_id': container.attr('dfo_id'),
            'dfmp_list': [],
        };

        if(dfm.dfo_id) {
            //get dfmp info
            container.find('.dfm-data-row').each(function(idx, row) {
                row = $(row);
                let dfmp = {
                    'param_i': idx,
                };

                if (row.find('[name=input_type]').length) {
                    dfmp.idf_type = row.find('[name=input_type]').find('option:selected').text();
                }

                if (row.find('[name=function]').length) {
                    let fn_id = row.find('[name=function]').find('option:selected').attr('fn_id');
                    dfmp.fn_id = fn_id ? fn_id : null;
                }

                if (row.find('[name=normalization]').length) {
                    dfmp.normalization = Number(row.find('[name=normalization]').find('option:selected').val());
                }

                dfm.dfmp_list.push(dfmp);
            });

            payload.dfm_list.push(dfm);
        }
    });

    // Multiple join
    if($('#join-function-select')) {
        let fn_id = $('#join-function-select option:selected').attr('fn_id');
        payload.multiplejoin_fn_id = fn_id ? fn_id : null;
    }

    function callback(data) {
        setTimeout(function(){
                $('.join-save').text('Save');
                $('.join-save').attr('disabled', false);
            },
            100);
    }
    mqtt_client.request('update_na', payload, callback);

}

function delete_join() {
    let na_id = $(this).attr('na_id');
    mqtt_client.request('delete_na', {'na_id': na_id}, reload_data);
}

function create_link(dfo_id, na_id) {
    function callback(data) {
        let na_id = data.na_id;
        mqtt_client.request('get_na_info', {'na_id': na_id, 'p_id': p_id}, show_join_management);
    }

    mqtt_client.request('create_link', {'na_id': na_id, 'dfo_id': dfo_id}, callback);
}

function delete_dfm() {
    let dfo_id = $(this).attr('dfo_id');
    let na_id = $('.join-delete').attr('na_id');
    function callback(data) {
        if (data) {
            mqtt_client.request('get_na_info', {'na_id': data.na_id, 'p_id': p_id}, show_join_management);
        } else {
            reload_data();
        }
    }

    mqtt_client.request('delete_link', {'na_id': na_id, 'dfo_id': dfo_id}, callback);
}

function get_new_join_container(dfo_id1, dfo_id2) {
    let dfo1_top = $('.dfo-container[dfo_id=' + dfo_id1 + ']').offset().top;
    let dfo2_top = $('.dfo-container[dfo_id=' + dfo_id2 + ']').offset().top;

    let join_container = null;

    let containers = $('.join-container.hidden-flag');
    join_container = containers.get(0);
    for (let i = 1; i < containers.length; ++i ) {
        if ($(containers.get(i)).offset().top > (dfo1_top + dfo2_top) / 2) {
            return join_container;
        }
        join_container = containers.get(i)
    }

    return join_container;
}

function bind_device() {
    let do_id = $(this).prev().children('img').attr('do_id');
    function callback(data) {
        if (!data || !data.device_list || data.device_list.length <= 0) {
            //NOTHING
        } else if (data.device_list.length == 1) {
            if (data.device_list[0].status == 'offline') {
                // offline device, not bind
            } else {
                mount_device(data.do_id, data.device_list[0].d_id);
            }
        } else {
            clear_right_window();
            let div_device_list = $('<div>', {'id': 'choose-device-list', 'do_id': do_id});
            for (let i = 0; i < data.device_list.length; ++i) {
                let device_container = $('<div>', {'class': 'choose-device', 'd_id': data.device_list[i].d_id});
                let device_label = $('<label>', {'class': 'choose-device-name', 'text': data.device_list[i].d_name});
                if (data.device_list[i].status == 'offline') {
                    device_container.addClass('offline');
                    device_label.addClass('offline');
                }
                device_container.append(device_label);
                div_device_list.append(device_container);
            }
            $('#right-window').append(div_device_list);
        }
    }

    if ($(this).children('span').hasClass('mounted')) {
        unmount_device(do_id);
    } else {
        mqtt_client.request('get_device_list', {'do_id': do_id}, callback);
    }
}

function click_device() {
    let d_id = $(this).attr('d_id');
    let do_id = $(this).parent().attr('do_id');
    if ($(this).children('label').hasClass('offline')){
        // offline device, reject bind
    } else {
        mount_device(do_id, d_id);
    }
}

function mount_device(do_id, d_id) {
    mqtt_client.request('bind_device', {'do_id': do_id, 'd_id': d_id}, reload_data);
}

function unmount_device(do_id) {
    mqtt_client.request('unbind_device', {'do_id': do_id}, reload_data);
}

//TODO function isDirty
function select_function() {
    if ('add new function' == $(this).find('option:selected').text()) {
        function _callback(data) {
            $('.df-manage-container, .join-manage-container').addClass('disappear-flag');
            $('#right-window').append(make_function_management_html(data));

            // code mirror
            window.myCodeMirror = CodeMirror.fromTextArea($('#function-code-area')[0], {
                'value': $('#function-code-area')[0],
                'mode': 'python',
                'indentUnit': 4,
                'lineWrapping': true,
                'lineNumbers': true,
                'styleActiveLine': true,
                'matchBrackets': true,

                'extraKeys': {
                    // replace 4 space instead of tab
                    'Tab': function(cm) {
                        cm.replaceSelection("    ", "end");
                    },
                }
            });
        }

        let dfo_id = $(this).parents('[dfo_id]').attr('dfo_id');
        mqtt_client.request('get_dfo_function_info', {'dfo_id': dfo_id}, _callback);
    }
}

function select_global_function() {
    $('#move-in-function').removeAttr('disabled');
    $('#move-out-function').attr('disabled', '');
    $('#df-function-select').find(':selected').removeAttr('selected');

    let fn_id = $('#global-function-select option:selected').attr('fn_id');
    let fn_name = $('#global-function-select option:selected').text();
    get_function_code(fn_id, fn_name);
}

function select_df_function() {
    $('#move-in-function').attr('disabled', '');
    $('#move-out-function').removeAttr('disabled');
    $('#global-function-select').find(':selected').removeAttr('selected');

    let fn_id = $('#df-function-select option:selected').attr('fn_id');
    let fn_name = $('#df-function-select option:selected').text();
    get_function_code(fn_id, fn_name);
}

function create_functionSDF() {
    let df_id = $(this).parents('[df_id]').attr('df_id');
    let fn_id = $('#global-function-select option:selected').attr('fn_id');

    if (!df_id) {df_id = null;}
    mqtt_client.request('create_functionSDF',
                        {'df_id': df_id, 'fn_id': fn_id},
                        reload_function_manage);
}

function delete_functionSDF() {
    let df_id = $(this).parents('[df_id]').attr('df_id');
    let fn_id = $('#df-function-select option:selected').attr('fn_id');

    if (!df_id) {df_id = null;}
    mqtt_client.request('delete_functionSDF',
                        {'df_id': df_id, 'fn_id': fn_id},
                        reload_function_manage);
}

function add_new_function() {
    get_function_code(null, '');
    $('#function-name').removeClass('readonly-flag');
    $('#function-name').removeAttr('readonly');
    $('#df-function-select').find(':selected').removeAttr('selected');
    $('#global-function-select').find(':selected').removeAttr('selected');
}

function save_function() {
    let df_id = $('.function-manage-container').attr('df_id');
    let fn_id = $('#function-name').attr('fn_id');
    let fn_name = $('#function-name').val().trim();
    let code = myCodeMirror.getValue();

    if (!fn_name) {
        alert('Function name can not empty.');
        return;
    }

    if (!fn_name.match(/^[a-zA-Z]/)) {
        alert('Function name should start with alphabet.')
        return
    }

    if (!df_id) { df_id = null;}

    if (fn_id) {
        mqtt_client.request('update_function',
                            {'df_id': df_id,
                             'fn_id': fn_id,
                             'code': code,},
                            reload_function_manage)
    } else {
        mqtt_client.request('create_function',
                            {'df_id': df_id,
                             'fn_name': fn_name,
                             'code': code,},
                            reload_function_manage)
    }
}

function delete_function() {
    let fn_id = $('#function-name').attr('fn_id');
    function callback(data) {
        reload_function_manage();
    }
    if (fn_id) {
        mqtt_client.request('delete_function', {'fn_id': fn_id}, callback);
    }
}

function close_manage_function() {
    function callback(data) {
        if (data['dfo_id']) { // update df function list
            let dfo_id = data['dfo_id'];
            let container = $('.df-manage-container[dfo_id='+dfo_id+'], .dfm-container[dfo_id='+dfo_id+']');
            $.each(container.find('.dfm-select[name=function], .df-func-select'), function (idx, select) {
                select = $(select);
                select.empty();

                select.append($('<option>', {'fn_id': 0, 'text': 'add new function'}));
                select.append($('<option>', {'fn_id': null, 'text': 'disable', 'selected': true}));
                $.each(data.fn_list, function(fn_idx, fn_info) {
                    let option = $('<option>', {'fn_id': fn_info.fn_id, 'text': fn_info.fn_name});
                    if (fn_info.fn_id == data.dfm_fn_list[idx]) {
                        option.attr('selected', true);
                    }
                    select.append(option);
                })
            });
        } else { // update join function list
            select = $('#join-function-select');
            select.empty();

            select.append($('<option>', {'fn_id': 0, 'text': 'add new function'}));
            select.append($('<option>', {'fn_id': null, 'text': 'disable', 'selected': true}));
            $.each(data.fn_list, function(fn_idx, fn_info) {
                let option = $('<option>', {'fn_id': fn_info.fn_id, 'text': fn_info.fn_name});
                if (fn_info.fn_id == data.multiplejoin_fn_id) {
                    option.attr('selected', true);
                }
                select.append(option);
            })
        }

    }
    let dfo_id = $(this).parents('[dfo_id]').attr('dfo_id');
    let na_id = $('.join-delete').attr('na_id');

    $('.function-manage-container').remove();
    $('.df-manage-container, .join-manage-container').removeClass('disappear-flag');
    mqtt_client.request('get_dfo_function_info', {'dfo_id': dfo_id, 'na_id': na_id}, callback);
}

function get_function_code(fn_id, fn_name) {
    function callback(data) {
        myCodeMirror.setValue(data.code);
    }

    $('#function-name').addClass('readonly-flag');
    $('#function-name').attr('readonly', '');
    $('#function-name').attr('fn_id', fn_id).val(fn_name);

    if (fn_id) {
        mqtt_client.request('get_function_info', {'fn_id': fn_id}, callback);
    } else {
        myCodeMirror.setValue('def run(*args):\n    return args[0]');
    }
}

function reload_function_manage() {
    function callback(data) {
        myCodeMirror.setValue('');
        $('#function-name').addClass('readonly-flag');
        $('#function-name').attr('readonly', '');
        $('#function-name').attr('fn_id', '').val('');

        let global_fn_select = $('#global-function-select');
        global_fn_select.empty();
        $.each(data.other_fn_list, function(index, fn_info) {
            global_fn_select.append($('<option>', {'fn_id': fn_info.fn_id, 'text': fn_info.fn_name}));
        });

        let df_fn_select = $('#df-function-select');
        df_fn_select.empty();
        $.each(data.fn_list, function(index, fn_info) {
            df_fn_select.append($('<option>', {'fn_id': fn_info.fn_id, 'text': fn_info.fn_name}));
        });
    }

    let dfo_id = $('.function-manage-container').attr('dfo_id');
    mqtt_client.request('get_dfo_function_info', {'dfo_id': dfo_id}, callback);
}

function show_save_gif (after_object, id='save-gif') {
    $('#'+id).remove()
    var img = $('<img>', {
        'src': '/static/images/save.gif',
        'id': id,
    });
    $(img).insertAfter(after_object);
    $(img).fadeIn(300).delay(500).fadeOut(500);
}

function resize() {
    $('#left-window').attr('style', 'min-height:' + $('#background').height() + 'px;');
    $('#right-window').attr('style', 'min-height:' + $('#background').height() + 'px;');

    let wid = $('.do-setting').width();
    $('.do-header').css('height', wid + 'px');
    $('.do-setting').css('height', wid + 'px');
    $('.do-setting-img').css('height', wid + 'px');
    $('.do-device').css('height', wid + 'px');
    $('.device-name').css('line-height', wid + 'px');
    $('.dfo-container').css('height', wid + 'px');
    $('.dfo-image').css('height', wid - 4  + 'px');
    $('.dfo-name').css('line-height', wid - 4  + 'px');
    $('#join-column').css('top', wid - 15 + 'px');
}

function redraw_connect_line(data) {
    resize();
    $('.join-container.hidden-flag').find('.join-clicked').removeClass('join-clicked');

    function callback(data) {
        let canvas = $('canvas');
        canvas.attr('width', canvas.width());
        canvas.attr('height', canvas.height());
        let ctx = canvas[0].getContext("2d");
        ctx.clearRect(0 , 0 , canvas.width(), canvas.height());
        $.each(data['na'], function(index, na) {
            let na_id = na['na_id'];
            let join = $($('.join-container').get(na.na_idx));
            join.removeClass('hidden-flag');
            join.attr('na_id', na.na_id);
            join.find('.join-title').text(na.na_name);

            $.each(na['input'], function(index, dfm) {
                draw_connect_line(na_id, dfm.dfo_id, dfm.df_type, dfm.color);
            });

            $.each(na['output'], function(index, dfm) {
                draw_connect_line(na_id, dfm.dfo_id, dfm.df_type, dfm.color);
            })
        });
    }
    if (data) {
        callback(data);
    } else if (p_id > 0){ // Do not get na list if p_id is less than to equal to 0
        mqtt_client.request('get_na_list', {'p_id': p_id}, callback);
    }
}

function draw_connect_line(na_id, dfo_id, df_type, color) {
    let join_point = get_join_draw_point(na_id);
    let dfo_point = get_dfo_draw_point(dfo_id, df_type);
    draw_line(dfo_point, join_point, color);
}

function get_join_draw_point(na_id) {
    let join_container = $('.join-container[na_id=' + na_id + ']');

    point = {}
    point.x = Number(join_container.find('img').offset().left)
            + Number(join_container.find('img').width()) / 2;

    point.y = Number($('#join-column').position().top)
            + Number(join_container.find('img').position().top)
            + Number(join_container.find('img').height()) / 2;

    return point;
}

function get_dfo_draw_point(dfo_id, df_type) {
    let dfo_container = $('.dfo-container[dfo_id=' + dfo_id + ']');

    point = {}
    if (df_type == 'input') {
        point.x = Number($('#in-device-column').position().left)
                + Number($('#in-device-column').outerWidth());
    } else {
        point.x  = Number($('#out-device-column').position().left);
    }
    point.y = Number(dfo_container.position().top)
            + Number(dfo_container.outerHeight()) / 2
            + Number($('#out-device-column').position().top);

    return point;
}

function draw_line(start_point, end_point, color='black') {
    let ctx = document.getElementById("background").getContext("2d");
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineCap = 'round';
    ctx.shadowBlur = 0;
    ctx.shadowColor = color;
    ctx.lineWidth = 1.5;
    ctx.moveTo(start_point.x, start_point.y);
    ctx.lineTo(end_point.x, end_point.y);
    ctx.stroke();
}

function init_subsystem_dm_nav(_, el) {
    /*
      * Assume the add new endpoint is triggered by HTTP PUT.
      */
    const self = $(el);
    const dmtype = self.data('dmtype');
    const da_endpoint = self.data('da-endpoint');
    const gui_endpoint = self.data('gui-endpoint');

    if (dmtype === undefined) return;

    fetch_subsystem_dm_list(self, `${da_endpoint}/`)

    self.on('click', 'ul > .dm-add-new', () => {
        const name = prompt(`Enter ${dmtype} Name`);
        if (name === null) return;
        $.ajax({
            url: `${da_endpoint}/`,
            type: 'PUT',
            async: false,
            contentType: 'application/json',
            dataType: 'json',
            data: JSON.stringify({ name: name }),
            success: function(res) {
                window.open(gui_endpoint);
                fetch_subsystem_dm_list(self, `${da_endpoint}/`)
            },
            error: function(jqXHR, res) {
                const msg = jqXHR.responseJSON.reason;
                alert(msg);
            }
        });
    });
}

function fetch_subsystem_dm_list(el, endpoint) {
    return $.ajax({
        url: endpoint,
        type: 'GET',
        async: true,
        success: function(res) {
            console.log(res);
            const ul = el.children('ul');
            ul.children(':not(.dm-add-new)').remove();
            res.forEach((i) => {
                // TODO: add dm_id
                ul.append(`<li class="dm-select"><a>${i.name}</a></li>`);
            });
        },
    });
}
