$(() => {
    const transitionStr = (initialZoom) => `translate(${initialZoom.x},${initialZoom.y}) scale(${initialZoom.scale})`;

    const margin = { top: 0, bottom: 0, left: 0, right: 0 };
    const width = 1100;
    const height = 500;
    const zoomInit = { x: -2000, y: -1100, scale: 9 }

    let lastZoomEvent = transitionStr(zoomInit);
    const limitZoom = (current, max, scale) => Math.min(0, Math.max(current, max - max * scale));
    const zoom = d3.zoom()
        .scaleExtent([1, 50])
        .on('zoom', () => {
            lastZoomEvent = d3.event.transform;
            const k = lastZoomEvent.k;
            lastZoomEvent.x = limitZoom(lastZoomEvent.x, width, k);
            lastZoomEvent.y = limitZoom(lastZoomEvent.y, height, k);
            svg.selectAll('#countries, circle, ellipse, #buoy-group, #boat-group').attr('transform', lastZoomEvent);
        });

    const svg = d3.select('#map')
        .append('svg')
        .classed('border', true)
        .attr('id', 'world')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .call(zoom);
        
    // initial zoom
    svg.call(zoom.transform, d3.zoomIdentity.translate(zoomInit.x, zoomInit.y).scale(zoomInit.scale));
    
    // projection takes [long, lat]
    const projection = d3.geoNaturalEarth1();
    const path = d3.geoPath(projection);

    const fetchAudioClips = (date, location, classification) => {
        const params = new URLSearchParams();
        params.set('month', date.getMonth()+1);
        params.set('day', date.getDate());
        params.set('hour', date.getUTCHours());
        if (location) {
            location = location.split(',');
            params.set('long', location[0]);
            params.set('lat', location[1]);
        }
        if (classification) params.set('classification', classification);

        $.ajax({
            success: (data) => {
                const $clips = $('#audio-clips').empty();
                
                if (data.length)
                    data.forEach(clip => $clips.append(`
                        <p class="audio-clip" data-audio-id="${clip.id}" data-audio-label="${clip.label}">
                            <i class="bi bi-play-circle"></i>
                            &#09;
                            ${clip.id}: ${clip.label}
                        </p>
                    `));
                else
                    $clips.append('<p>No clips found for the current time and buoy!</p>')
            },
            url: `/audio_clips/?${params.toString()}`
        });
    };

    const fetchBoatLocations = (boatType, date) => {
        const params = new URLSearchParams();
        params.set('month', date.getMonth()+1);
        params.set('day', date.getDate());
        params.set('hour', date.getUTCHours());
        if (boatType) params.set('boat_type', boatType);
        
        $.ajax({
            success: (data) => {
                $('#filter').prop('disabled', false);
                $('#loading').addClass('d-none');

                data = Array.from(d3.group(data, d => d.mmsi).values());
                svg.select('#boat-group').remove();
                const boatGroup = svg.append('g')
                    .attr('id', 'boat-group')
                    .attr('transform', lastZoomEvent);
                boatGroup.selectAll('.boat')
                    .data(data)
                    .enter()
                    .append('g')
                    .classed('boat-svg', true)
                    .attr('transform', d => transitionStr({x: projection([d[0].long, d[0].lat])[0], y: projection([d[0].long, d[0].lat])[1], scale: 1}))
                    .append('svg')
                    .attr('viewBox', '0 0 512 512')
                    .attr('x', -300.75)
                    .attr('y', -.8)
                    .attr('class', d => `boat ${d[0].boat_type.replace(/[\s/]/, '-')}`)
                    .append('path')
                    .attr('d', 'M410.866,181.063A32.1,32.1,0,0,0,380.793,160H341.554l-16-96H221.112l16,96H197.554l-16-96H77.112l16,96H48v88H16V448H443.727L496,282.466V248H435.207ZM298.446,96l10.667,64H269.554L258.888,96Zm-144,0,10.667,64H125.554L114.888,96ZM80,192H380.793l20.363,56H80Zm383.222,88L420.273,416H48V280Z')
                    .attr('transform', 'scale(.003)')
                    
                // svg.selectAll('ellipse')
                //     .data(data)
                //     .enter()
                //     .append('ellipse')
                //     .classed('boat', true)
                //     .attr('cx', d => projection([d[0].long, d[0].lat])[0])
                //     .attr('cy', d => projection([d[0].long, d[0].lat])[1])
                //     .attr('rx', 1)
                //     .attr('ry', .5)
                //     .attr('transform', lastZoomEvent);
            },
            url: `/ais/?${params.toString()}`
        });
    };

    const fetchBuoyLocations = () => {
        $.ajax({
            method: 'GET',
            success: (data) => {
                if (data.length) buoyClicked = data[0];

                data.forEach((buoy, i) => $('#buoy').append(`<option value="${buoy.long},${buoy.lat}">Buoy ${i + 1}</option`))

                const buoys = svg.append('g')
                    .attr('id', 'buoy-group')
                    .attr('transform', lastZoomEvent);
                buoys.selectAll('.buoy')
                    .data(data)
                    .enter()
                    .append('g')
                    .classed('buoy-svg', true)
                    .attr('transform', d => transitionStr({x: projection([d.long, d.lat])[0], y: projection([d.long, d.lat])[1], scale: .1}))
                    .append('svg')
                    .attr('viewBox', '0 0 512 512')
                    .attr('x', -310)
                    .attr('y', -5)
                    .append('path')
                    .classed('buoy', true)
                    .attr('d', 'M418.133,409.6h-10.402l-64.563-258.244c8.96-3.985,15.232-12.937,15.232-23.356v-17.067c0-14.114-11.486-25.6-25.6-25.6\
                        h-68.267V67.055c14.677-3.814,25.6-17.067,25.6-32.922C290.133,15.309,274.825,0,256,0s-34.133,15.309-34.133,34.133\
                        c0,15.855,10.923,29.107,25.6,32.922v18.278H179.2c-14.114,0-25.6,11.486-25.6,25.6V128c0,10.428,6.263,19.413,15.224,23.398\
                        l-57.634,230.528c-1.143,4.574,1.638,9.207,6.204,10.351c0.7,0.179,1.391,0.256,2.082,0.256c3.823,0,7.296-2.586,8.269-6.46\
                        l23.987-95.94h95.735V384c0,4.71,3.823,8.533,8.533,8.533s8.533-3.823,8.533-8.533v-93.867h95.735l32.521,130.074\
                        c0.947,3.789,4.361,6.46,8.277,6.46h17.067c4.702,0,8.533,3.831,8.533,8.533c0,32.939-26.795,59.733-59.733,59.733H145.067\
                        c-32.939,0-59.733-26.795-59.733-59.733c0-4.702,3.831-8.533,8.533-8.533h273.067c4.71,0,8.533-3.823,8.533-8.533\
                        s-3.823-8.533-8.533-8.533H93.867c-14.114,0-25.6,11.486-25.6,25.6c0,42.342,34.458,76.8,76.8,76.8h221.867\
                        c42.342,0,76.8-34.458,76.8-76.8C443.733,421.086,432.247,409.6,418.133,409.6z M238.933,34.133\
                        c0-9.412,7.654-17.067,17.067-17.067s17.067,7.654,17.067,17.067c0,9.412-7.654,17.067-17.067,17.067\
                        S238.933,43.546,238.933,34.133z M213.871,136.533c-4.719,0-8.542,3.823-8.542,8.533c0,4.71,3.823,8.533,8.542,8.533h33.596\
                        v119.467h-91.469l31.479-125.926c0.64-2.551,0.06-5.257-1.553-7.322c-1.613-2.074-4.096-3.285-6.724-3.285\
                        c-4.702,0-8.533-3.831-8.533-8.533v-17.067c0-4.702,3.831-8.533,8.533-8.533h153.6c4.702,0,8.533,3.831,8.533,8.533V128\
                        c0,4.702-3.831,8.533-8.533,8.533H213.871z M264.533,273.067V153.6h61.602l29.867,119.467H264.533z')
                    .attr('transform', 'scale(.05)')

                // const buoys = svg.append('g');
                // svg.selectAll('rect')
                //     .data(data)
                //     .enter()
                //     .append('rect')
                //     .attr('x', d => projection([d.long, d.lat])[0])
                //     .attr('y', d => projection([d.long, d.lat])[1])
                //     .attr('width', .5)
                //     .attr('height', 2)
                //     .attr('transform', lastZoomEvent)
                //     .on('click', (d) => buoyClicked = d);
            },
            url: '/buoys/'
        })
    };

    const plotMap = (world) => {
        const paths = (svg.select('#countries').node()
                ? svg.select('#countries')
                : svg.append('g')
                    .attr('id', 'countries')
            ).selectAll('path')
            .data(world.features);
    
        paths.enter()
            .append('path')
            .attr('d', path)
            .merge(paths)
            .attr('fill', 'darkseagreen')
            .attr('stroke', 'white')
            .attr('stroke-width', '.05px');
    
        svg.selectAll('#countries, circle, ellipse').attr('transform', lastZoomEvent);
        fetchBuoyLocations();
    };

    const postAudioFile = (file) => {
        const form = new FormData();
        form.append('audio', file);

        $.ajax({
            contentType: false,
            data: form,
            processData: false,
            success: (data) => {
                $('#results').text(`The uploaded audio matches with a ${data.name}!`);
                $('#modal').modal('show');
            },
            type: 'POST',
            url: 'predict/'
        });
    };

    const search = (date) => {
        $('#filter').attr('disabled', true);
        $('#loading').removeClass('d-none');
        fetchBoatLocations($('#boat').val(), date);
        fetchAudioClips(date, $('#buoy').val(), $('#classification').val());
    };

    const setDate = (date) => $('#date-val')
        .data('date', date)
        .text(date.toISOString().replace('T', ' ').slice(0, 16));


    Promise.all([
        d3.json('static/world_countries.json')
    ]).then(
        values => plotMap(values[0])
    );
    
    const wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: 'lightblue',
        progressColor: 'purple',
        normalize: true,
        plugins: [
            WaveSurfer.spectrogram.create({
                container: "#wave-spectrogram",
                fftSamples: 512,
                labels: true
            })
        ]
    });
    wavesurfer.on('ready', () => $('#wave-spectrogram').find('canvas').not('.spec-labels').css('position', 'unset'));
    
    const buoyLocations = {
        1: '-78.37406667,32.0705',
        2: '-75.02038,35.19955',
        3: '-80.00312,30.49274'
    }
    const initialDate = new Date('2018-01-01');
    [
        ['2018-06-15 23:00', 1],
        ['2018-06-24 10:00', 1],
        ['2018-06-24 12:00', 1],
        ['2018-10-05 08:00', 1],
    ].forEach(d => $('#quick-dates').append(`<p class="quick-date mb-1" data-date="${d[0]}" data-buoy-index="${d[1]}"><i class="bi bi-filter"></i>&#09;${d[0]}</p>`));

    $('#audio-clips').on('click', '.audio-clip', function() {
        const id = $(this).data('audio-id');
        const label = $(this).data('audio-label');
        const url = `/audio_file/?id=${id}&label=${label}`;

        $.ajax({
            error: () => alert('No audio found!'),
            success: () => {
                wavesurfer.load(url);
                $('#play').removeClass('d-none');
            },
            url: url
        });
        
    });
    $('#date').slider({
        min: initialDate.getTime() / 1000,
        max: new Date('2018-12-31').getTime() / 1000,
        value: initialDate.getTime() / 1000,
        step: 60 * 60,
        slide: (e, ui) => setDate(new Date(ui.value * 1000))
    });
    $('#date-val').data('date', initialDate).text(initialDate.toISOString().replace('T', ' ').slice(0, 16));
    $('#filter').on('click', () => search($('#date-val').data('date')));
    $('#play').on('click', wavesurfer.playPause.bind(wavesurfer));
    $('#quick-dates').on('click', '.quick-date', function() {
        const date = new Date($(this).data('date'));
        const dateUtc = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate(), date.getHours()));
        setDate(dateUtc);
        $('#date').slider('value', dateUtc.getTime() / 1000);
        $(`option[value="${buoyLocations[$(this).data('buoy-index')]}"]`).prop('selected', true);
        $('#classification > option[value="unknown"]').prop('selected', true);
        search(dateUtc);
    });
    $('#upload-btn').on('click', () => $('#upload').click());
    $('#upload').on('change', (e) => {
        postAudioFile(e.target.files[0]);
        e.target.value = null;
    });
});
